"""Tests for the Androguard-based feature extractor.

All APKs are built by tests/fixtures/apk_builder.py (benign, synthetic, test-only).
"""

import hashlib
import os
import zipfile

import pytest

from app.ml.schema import DEX_FEATURES, FEATURE_NAMES, MANIFEST_FEATURES
from app.services import apk_features as af
from app.services.apk_features import (
    ExtractionLimits,
    IsolatedExtractor,
    extract_features,
    extract_features_isolated,
)
from tests.fixtures.apk_builder import (
    DEFAULT_PACKAGE,
    MAIN_ACTIVITY,
    build_axml_manifest,
    build_dex,
)

PAD = b"padding text so the archive is above the minimum size " * 40


def stored_zip(path, entries):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        for name, blob in entries.items():
            zf.writestr(name, blob)
    return path


def good_manifest():
    return build_axml_manifest(DEFAULT_PACKAGE, ["android.permission.INTERNET"], [MAIN_ACTIVITY], [], [])


def good_dex():
    return build_dex(["L" + MAIN_ACTIVITY.replace(".", "/") + ";"])


def assert_no_invented_values(result):
    assert all(v is None for v in result.features.values()), result.features


# --------------------------------------------------------------------------
# Valid APK
# --------------------------------------------------------------------------
def test_valid_apk_exact_feature_values(make_apk):
    path = make_apk(
        permissions=[
            "android.permission.INTERNET",
            "android.permission.INTERNET",  # duplicate must count once
            "android.permission.CAMERA",  # dangerous
            "com.vendor.permission.CUSTOM",  # unknown, not dangerous
        ],
        activities=[MAIN_ACTIVITY, DEFAULT_PACKAGE + ".Second"],
        services=[".SyncService"],  # relative name
        receivers=[DEFAULT_PACKAGE + ".BootReceiver"],
        method_refs=[
            ("Landroid/app/Activity;", "onCreate", "V", ("Landroid/os/Bundle;",)),
            ("Landroid/util/Log;", "d", "I", ("Ljava/lang/String;", "Ljava/lang/String;")),
            ("Landroid/telephony/SmsManager;", "sendTextMessage", "V", ("Ljava/lang/String;",)),
            ("Ljava/lang/Runtime;", "exec", "Ljava/lang/Process;", ("Ljava/lang/String;",)),
            # reference to a class defined inside this DEX: internal, must NOT count
            ("L" + MAIN_ACTIVITY.replace(".", "/") + ";", "helper", "V", ()),
        ],
    )
    result = extract_features(path)

    assert result.status == "OK"
    assert result.error_code is None
    assert result.package_name == DEFAULT_PACKAGE
    assert result.features == {
        "permissions_count": 3,
        "activities_count": 2,
        "services_count": 1,
        "receivers_count": 1,
        "api_calls_count": 4,  # 5 references minus 1 internal
        "strings_count": 19,  # 10 type descriptors + 5 names + 3 new shorty strings + 1 extra
        "dangerous_permissions": 1,
        "suspicious_api_calls": 2,  # SmsManager.sendTextMessage, Runtime.exec
    }
    assert result.indicators["dangerous_permission_names"] == ["android.permission.CAMERA"]
    assert result.indicators["suspicious_api_matches"] == [
        "Ljava/lang/Runtime;.exec",
        "Landroid/telephony/SmsManager;.sendTextMessage",
    ] or sorted(result.indicators["suspicious_api_matches"]) == sorted(
        ["Ljava/lang/Runtime;.exec", "Landroid/telephony/SmsManager;.sendTextMessage"]
    )


def test_result_has_hash_size_and_version(valid_apk):
    result = extract_features(valid_apk)
    assert result.sha256 == hashlib.sha256(valid_apk.read_bytes()).hexdigest()
    assert result.file_size == valid_apk.stat().st_size
    assert result.extractor_version.startswith("1.0.0+perm")


def test_extraction_is_deterministic(valid_apk):
    first, second = extract_features(valid_apk), extract_features(valid_apk)
    assert first.features == second.features
    assert first.sha256 == second.sha256


def test_all_eight_features_present_and_integer(valid_apk):
    result = extract_features(valid_apk)
    assert tuple(result.features) == FEATURE_NAMES
    assert all(isinstance(v, int) and v >= 0 for v in result.features.values())


def test_zero_is_reported_as_zero_for_a_real_apk(make_apk):
    """A genuine count of zero is data; it must not be confused with 'unknown' (None)."""
    result = extract_features(make_apk(permissions=[], activities=[], method_refs=[], extra_strings=[], defined_classes=["Lcom/x/A;"]))
    assert result.status == "OK"
    assert result.features["permissions_count"] == 0
    assert result.features["services_count"] == 0
    assert result.features["suspicious_api_calls"] == 0


def test_multi_dex_union(make_apk):
    second = build_dex(
        ["Lcom/example/guardian/testapp/Other;"],
        [
            ("Landroid/util/Log;", "d", "I", ("Ljava/lang/String;", "Ljava/lang/String;")),  # already in dex 1
            ("Landroid/os/Handler;", "post", "Z", ("Ljava/lang/Runnable;",)),
        ],
        ["extra string"],
    )
    result = extract_features(make_apk(extra_dex_files=[second]))
    assert result.status == "OK"
    assert result.indicators["dex_file_count"] == 2
    assert result.features["api_calls_count"] == 3  # onCreate, Log.d (shared), Handler.post
    assert result.features["strings_count"] == 20  # 13 + 13 minus 6 strings present in both


# --------------------------------------------------------------------------
# Invalid and corrupted inputs: FAILED, error code, and no fabricated values
# --------------------------------------------------------------------------
def test_missing_file(tmp_path):
    result = extract_features(tmp_path / "nope.apk")
    assert (result.status, result.error_code) == ("FAILED", af.FILE_NOT_FOUND)
    assert_no_invented_values(result)
    assert result.sha256 is None


def test_empty_file(tmp_path):
    path = tmp_path / "empty.apk"
    path.write_bytes(b"")
    result = extract_features(path)
    assert (result.status, result.error_code) == ("FAILED", af.FILE_TOO_SMALL)
    assert_no_invented_values(result)


def test_tiny_file(tmp_path):
    path = tmp_path / "tiny.apk"
    path.write_bytes(b"PK\x03\x04" + b"\x00" * 60)
    result = extract_features(path)
    assert result.error_code == af.FILE_TOO_SMALL
    assert_no_invented_values(result)


def test_text_file_renamed_to_apk(tmp_path):
    path = tmp_path / "notes.apk"
    path.write_text("this is not an apk. " * 100)
    result = extract_features(path)
    assert (result.status, result.error_code) == ("FAILED", af.NOT_A_ZIP)
    assert_no_invented_values(result)
    assert result.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()  # hash still reported


def test_zip_signature_followed_by_garbage(tmp_path):
    path = tmp_path / "garbage.apk"
    path.write_bytes(b"PK\x03\x04" + os.urandom(3000))
    result = extract_features(path)
    assert (result.status, result.error_code) == ("FAILED", af.CORRUPT_ZIP)
    assert_no_invented_values(result)


def test_truncated_apk(valid_apk, tmp_path):
    data = valid_apk.read_bytes()
    path = tmp_path / "truncated.apk"
    path.write_bytes(data[: int(len(data) * 0.6)])
    assert path.stat().st_size >= 512  # make sure we exercise the zip check, not the size check
    result = extract_features(path)
    assert (result.status, result.error_code) == ("FAILED", af.CORRUPT_ZIP)
    assert_no_invented_values(result)


def test_apk_without_manifest(tmp_path):
    path = stored_zip(tmp_path / "nomanifest.apk", {"classes.dex": good_dex(), "assets/pad.txt": PAD})
    result = extract_features(path)
    assert (result.status, result.error_code) == ("FAILED", af.MISSING_MANIFEST)
    assert_no_invented_values(result)


def test_garbage_manifest(tmp_path):
    path = stored_zip(
        tmp_path / "badmanifest.apk",
        {"AndroidManifest.xml": b"\x03\x00\x08\x00garbage" * 200, "classes.dex": good_dex()},
    )
    result = extract_features(path)
    assert (result.status, result.error_code) == ("FAILED", af.MANIFEST_PARSE_ERROR)
    assert_no_invented_values(result)


def test_bit_flip_in_manifest_is_detected_by_crc(tmp_path):
    manifest = good_manifest()
    path = stored_zip(tmp_path / "flip.apk", {"AndroidManifest.xml": manifest, "classes.dex": good_dex(), "assets/pad.txt": PAD})
    data = bytearray(path.read_bytes())
    pos = bytes(data).find(manifest)
    assert pos > 0
    data[pos + len(manifest) // 2] ^= 0xFF
    path.write_bytes(bytes(data))
    result = extract_features(path)
    assert (result.status, result.error_code) == ("FAILED", af.CORRUPT_ZIP)
    assert_no_invented_values(result)


def test_apk_without_dex_is_partial_not_zero_filled(tmp_path):
    path = stored_zip(tmp_path / "nodex.apk", {"AndroidManifest.xml": good_manifest(), "assets/pad.txt": PAD})
    result = extract_features(path)
    assert (result.status, result.error_code) == ("PARTIAL", af.MISSING_DEX)
    for name in DEX_FEATURES:
        assert result.features[name] is None  # unknown, not 0
    for name in MANIFEST_FEATURES:
        assert isinstance(result.features[name], int)


def test_dex_with_bad_magic_is_partial(tmp_path):
    path = stored_zip(
        tmp_path / "baddex.apk",
        {"AndroidManifest.xml": good_manifest(), "classes.dex": b"NOTDEX\n" + b"\x00" * 800},
    )
    result = extract_features(path)
    assert (result.status, result.error_code) == ("PARTIAL", af.DEX_PARSE_ERROR)
    for name in DEX_FEATURES:
        assert result.features[name] is None
    assert result.features["permissions_count"] == 1


def test_truncated_dex_is_partial(tmp_path):
    path = stored_zip(
        tmp_path / "truncdex.apk",
        {"AndroidManifest.xml": good_manifest(), "classes.dex": good_dex()[:200], "assets/pad.txt": PAD},
    )
    result = extract_features(path)
    assert (result.status, result.error_code) == ("PARTIAL", af.DEX_PARSE_ERROR)
    for name in DEX_FEATURES:
        assert result.features[name] is None


def test_bit_flip_in_dex_is_partial(tmp_path):
    dex = good_dex()
    path = stored_zip(tmp_path / "flipdex.apk", {"AndroidManifest.xml": good_manifest(), "classes.dex": dex, "assets/pad.txt": PAD})
    data = bytearray(path.read_bytes())
    pos = bytes(data).find(dex)
    assert pos > 0
    data[pos + len(dex) // 2] ^= 0xFF
    path.write_bytes(bytes(data))
    result = extract_features(path)
    assert result.status == "PARTIAL"
    assert result.error_code == af.DEX_PARSE_ERROR
    for name in DEX_FEATURES:
        assert result.features[name] is None


# --------------------------------------------------------------------------
# Resource limits
# --------------------------------------------------------------------------
def test_oversized_file_rejected(valid_apk):
    result = extract_features(valid_apk, ExtractionLimits(max_size_bytes=600))
    assert result.error_code == af.FILE_TOO_LARGE
    assert_no_invented_values(result)


def test_too_many_zip_entries_rejected(valid_apk):
    result = extract_features(valid_apk, ExtractionLimits(max_zip_entries=1))
    assert result.error_code == af.ZIP_LIMITS_EXCEEDED
    assert_no_invented_values(result)


def test_declared_expansion_limit_rejected(valid_apk):
    result = extract_features(valid_apk, ExtractionLimits(max_total_uncompressed_bytes=100))
    assert result.error_code == af.ZIP_LIMITS_EXCEEDED
    assert_no_invented_values(result)


def test_extractor_never_raises_on_arbitrary_bytes(tmp_path):
    for i, blob in enumerate([b"\x00" * 2000, b"PK\x03\x04" * 500, os.urandom(2000), b"PK\x05\x06" + b"\x00" * 600]):
        path = tmp_path / f"x{i}.apk"
        path.write_bytes(blob)
        result = extract_features(path)
        assert result.status == "FAILED"
        assert_no_invented_values(result)


# --------------------------------------------------------------------------
# Process isolation and timeout
# --------------------------------------------------------------------------
def test_isolated_matches_in_process(valid_apk):
    direct = extract_features(valid_apk)
    isolated = extract_features_isolated(valid_apk, timeout_seconds=120)
    assert isolated.status == "OK"
    assert isolated.features == direct.features
    assert isolated.sha256 == direct.sha256


def test_isolated_timeout_then_recovers(valid_apk):
    with IsolatedExtractor(timeout_seconds=0.01) as extractor:
        timed_out = extractor.extract(valid_apk)  # child cannot even start in 10 ms
        assert (timed_out.status, timed_out.error_code) == ("FAILED", af.TIMEOUT)
        assert_no_invented_values(timed_out)
        assert timed_out.sha256 == hashlib.sha256(valid_apk.read_bytes()).hexdigest()
        extractor.timeout_seconds = 120
        recovered = extractor.extract(valid_apk)  # a fresh worker is started
        assert recovered.status == "OK"


def test_isolated_reports_failures_without_crashing(tmp_path):
    path = tmp_path / "notes.apk"
    path.write_text("not an apk " * 100)
    result = extract_features_isolated(path, timeout_seconds=120)
    assert result.error_code == af.NOT_A_ZIP
    assert_no_invented_values(result)


# --------------------------------------------------------------------------
# Regression tests for cross-version problems (found with Androguard 3.3.5)
# --------------------------------------------------------------------------
def test_component_names_are_qualified_and_deduplicated(make_apk):
    """Androguard 3.x turned '.Svc' into 'pkg..Svc'; names must be normalised by our code."""
    result = extract_features(
        make_apk(
            services=[".Svc", DEFAULT_PACKAGE + ".Svc", "OtherSvc"],  # first two are the same class
            receivers=[".Rcv"],
            activities=[MAIN_ACTIVITY, ".MainActivity"],  # same class again
        )
    )
    assert result.status == "OK"
    assert result.features["services_count"] == 2
    assert result.features["activities_count"] == 1
    assert result.indicators["services"] == [DEFAULT_PACKAGE + ".OtherSvc", DEFAULT_PACKAGE + ".Svc"]
    every_name = [n for key in ("activities", "services", "receivers") for n in result.indicators[key]]
    assert all(".." not in n and not n.startswith(".") for n in every_name), every_name


def test_activity_alias_is_counted_as_an_activity(make_apk):
    result = extract_features(make_apk(activities=[MAIN_ACTIVITY], activity_aliases=[".Alias"]))
    assert result.features["activities_count"] == 2
    assert DEFAULT_PACKAGE + ".Alias" in result.indicators["activities"]


def test_androguard_version_is_recorded(valid_apk):
    result = extract_features(valid_apk)
    assert result.indicators["androguard_version"] == af.androguard_version()
    assert result.indicators["androguard_version"] not in ("", "unknown")


def test_unsupported_androguard_is_an_error_not_a_failed_sample(valid_apk, monkeypatch):
    def broken():
        raise af.ExtractorEnvironmentError("Androguard is not installed")

    monkeypatch.setattr(af, "load_androguard", broken)
    with pytest.raises(af.ExtractorEnvironmentError):
        extract_features(valid_apk)


def test_dex_validation_catches_damage_without_relying_on_androguard():
    import struct

    dex = good_dex()
    af.validate_dex_blob(dex)  # a well-formed DEX passes

    flipped = bytearray(dex)
    flipped[len(dex) // 2] ^= 0xFF
    with pytest.raises(ValueError, match="checksum"):
        af.validate_dex_blob(bytes(flipped))
    with pytest.raises(ValueError, match="declares"):
        af.validate_dex_blob(dex[:-4])
    with pytest.raises(ValueError, match="magic"):
        af.validate_dex_blob(b"NOTDEX\n" + dex[7:])
    with pytest.raises(ValueError, match="magic"):
        af.validate_dex_blob(dex[:40])
    bad_header = bytearray(dex)
    struct.pack_into("<I", bad_header, 36, 0x80)
    with pytest.raises(ValueError, match="header size"):
        af.validate_dex_blob(bytes(bad_header))
