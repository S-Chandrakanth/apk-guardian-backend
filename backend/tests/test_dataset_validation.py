import pandas as pd
import pytest

from app.ml.dataset_validation import (
    find_duplicate_hashes,
    parse_count,
    validate_manifest_file,
    validate_manifest_frame,
)
from app.ml.schema import FEATURE_NAMES, MANIFEST_COLUMNS
from tests.conftest import sha

VERSION = "1.0.0+perm1.0.0+api1.0.0"


def make_row(i, **overrides):
    row = {
        "sample_id": f"s{i}",
        "sha256": sha(f"sample-{i}"),
        "label": "MALWARE" if i % 2 else "BENIGN",
        "original_category": "Adware" if i % 2 else "Benign",
        "label_source": "unit test source",
        "extractor_version": VERSION,
        "extraction_status": "OK",
        **{name: str(i + 1) for name in FEATURE_NAMES},
        "error_code": "",
        "error_message": "",
    }
    row.update(overrides)
    return row


def frame(rows):
    return pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS))


def good_frame(n=6):
    return frame([make_row(i) for i in range(n)])


def reasons(result):
    return result.report.stats.get("rejected_by_reason", {})


# ---------------- happy path ----------------
def test_valid_manifest_passes():
    result = validate_manifest_frame(good_frame(), VERSION)
    assert result.report.ok, result.report.errors
    assert len(result.training_rows) == 6
    assert result.report.stats["label_counts"] == {"BENIGN": 3, "MALWARE": 3}
    assert all(str(result.training_rows[n].dtype) == "int64" for n in FEATURE_NAMES)
    assert result.rejected.empty


def test_input_frame_is_not_modified():
    df = good_frame()
    before = df.copy()
    validate_manifest_frame(df, VERSION)
    pd.testing.assert_frame_equal(df, before)


# ---------------- structure ----------------
def test_missing_columns_reported():
    df = good_frame().drop(columns=["label", "strings_count"])
    result = validate_manifest_frame(df)
    assert not result.report.ok
    assert "label" in result.report.errors[0] and "strings_count" in result.report.errors[0]
    assert result.training_rows.empty


def test_empty_manifest():
    result = validate_manifest_frame(frame([]))
    assert not result.report.ok


# ---------------- hashes ----------------
@pytest.mark.parametrize("bad", ["", "abc", "z" * 64, "a" * 63, "a" * 65])
def test_invalid_sha256_rejected(bad):
    result = validate_manifest_frame(frame([make_row(0, sha256=bad), *[make_row(i) for i in range(1, 6)]]), VERSION)
    assert reasons(result)["INVALID_SHA256"] == 1
    assert len(result.training_rows) == 5


def test_uppercase_sha256_is_normalised_not_rejected():
    result = validate_manifest_frame(frame([make_row(0, sha256=sha("sample-0").upper()), *[make_row(i) for i in range(1, 6)]]), VERSION)
    assert not reasons(result)
    assert result.training_rows.loc[0, "sha256"] == sha("sample-0")


def test_duplicates_detected_and_removed_by_sha256():
    rows = [make_row(i) for i in range(6)] + [make_row(99, sha256=sha("sample-2"), label="BENIGN")]
    result = validate_manifest_frame(frame(rows), VERSION)
    assert result.report.stats["duplicate_rows_removed"] == 1
    assert reasons(result)["DUPLICATE_SHA256"] == 1
    assert len(result.training_rows) == 6
    assert result.training_rows["sha256"].is_unique


def test_duplicate_keeps_the_successful_extraction():
    failed = make_row(50, sha256=sha("same"), label="BENIGN", extraction_status="FAILED",
                      error_code="CORRUPT_ZIP", **{n: "" for n in FEATURE_NAMES})
    ok = make_row(51, sha256=sha("same"), label="BENIGN")  # same hash AND same label
    result = validate_manifest_frame(frame([failed, ok, *[make_row(i) for i in range(6)]]), VERSION)
    kept = result.all_rows[result.all_rows["sha256"] == sha("same")]
    assert len(kept) == 1 and kept.iloc[0]["extraction_status"] == "OK"


def test_conflicting_labels_for_same_hash_are_excluded_entirely():
    rows = [make_row(i) for i in range(6)]
    rows += [make_row(60, sha256=sha("c"), label="BENIGN"), make_row(61, sha256=sha("c"), label="MALWARE")]
    result = validate_manifest_frame(frame(rows), VERSION)
    assert reasons(result)["CONFLICTING_LABELS"] == 2
    assert sha("c") not in set(result.all_rows["sha256"])
    assert any("conflicting labels" in w for w in result.report.warnings)


def test_find_duplicate_hashes_utility():
    assert find_duplicate_hashes([sha("a"), sha("b"), sha("a"), "bad", "bad"]) == {sha("a"): 2}
    assert find_duplicate_hashes([sha("a"), sha("a").upper()]) == {sha("a"): 2}
    assert find_duplicate_hashes([]) == {}


# ---------------- labels ----------------
def test_blank_label_is_rejected_never_assigned():
    result = validate_manifest_frame(frame([make_row(0, label=""), *[make_row(i) for i in range(1, 6)]]), VERSION)
    assert reasons(result)["UNLABELLED"] == 1
    assert sha("sample-0") not in set(result.all_rows["sha256"])


def test_unknown_label_rejected():
    result = validate_manifest_frame(frame([make_row(0, label="TROJAN"), *[make_row(i) for i in range(1, 6)]]), VERSION)
    assert reasons(result)["INVALID_LABEL"] == 1


def test_label_case_is_normalised():
    result = validate_manifest_frame(frame([make_row(0, label=" malware "), *[make_row(i) for i in range(1, 6)]]), VERSION)
    assert not reasons(result)
    assert result.training_rows.loc[0, "label"] == "MALWARE"


def test_missing_label_source_rejected():
    result = validate_manifest_frame(frame([make_row(0, label_source=""), *[make_row(i) for i in range(1, 6)]]), VERSION)
    assert reasons(result)["MISSING_LABEL_SOURCE"] == 1


def test_missing_sample_id_rejected():
    result = validate_manifest_frame(frame([make_row(0, sample_id=" "), *[make_row(i) for i in range(1, 6)]]), VERSION)
    assert reasons(result)["MISSING_SAMPLE_ID"] == 1


# ---------------- extraction status and feature values ----------------
def test_invalid_status_rejected():
    result = validate_manifest_frame(frame([make_row(0, extraction_status="DONE"), *[make_row(i) for i in range(1, 6)]]), VERSION)
    assert reasons(result)["INVALID_STATUS"] == 1


@pytest.mark.parametrize("bad", ["", "-1", "abc", "3.5", "1e3", "nan", "None"])
def test_ok_row_with_bad_feature_value_rejected(bad):
    result = validate_manifest_frame(frame([make_row(0, api_calls_count=bad), *[make_row(i) for i in range(1, 6)]]), VERSION)
    assert reasons(result)["BAD_FEATURE_VALUE"] == 1
    assert len(result.training_rows) == 5


def test_integer_written_as_float_text_is_accepted():
    result = validate_manifest_frame(frame([make_row(0, strings_count="5.0"), *[make_row(i) for i in range(1, 6)]]), VERSION)
    assert not reasons(result)
    assert result.training_rows.loc[0, "strings_count"] == 5


def test_zero_is_a_valid_feature_value():
    result = validate_manifest_frame(frame([make_row(0, services_count="0"), *[make_row(i) for i in range(1, 6)]]), VERSION)
    assert not reasons(result)


def test_failed_row_kept_out_of_training_but_not_deleted():
    failed = make_row(70, extraction_status="FAILED", error_code="CORRUPT_ZIP", **{n: "" for n in FEATURE_NAMES})
    result = validate_manifest_frame(frame([failed, *[make_row(i) for i in range(6)]]), VERSION)
    assert result.report.ok
    assert len(result.training_rows) == 6
    assert (result.all_rows["extraction_status"] == "FAILED").sum() == 1
    assert any("excluded from training" in w for w in result.report.warnings)


def test_failed_row_carrying_feature_values_is_rejected_as_fabricated():
    fake = make_row(71, extraction_status="FAILED", error_code="CORRUPT_ZIP")  # features still filled in
    result = validate_manifest_frame(frame([fake, *[make_row(i) for i in range(6)]]), VERSION)
    assert reasons(result)["FAILED_ROW_HAS_FEATURES"] == 1


def test_partial_row_with_blank_dex_features_is_kept_but_not_trained_on():
    partial = make_row(72, extraction_status="PARTIAL", error_code="MISSING_DEX",
                       api_calls_count="", strings_count="", suspicious_api_calls="")
    result = validate_manifest_frame(frame([partial, *[make_row(i) for i in range(6)]]), VERSION)
    assert result.report.ok
    assert len(result.training_rows) == 6
    assert (result.all_rows["extraction_status"] == "PARTIAL").sum() == 1


# ---------------- training-set level checks ----------------
def test_single_class_is_an_error():
    rows = [make_row(i, label="MALWARE") for i in range(6)]
    result = validate_manifest_frame(frame(rows), VERSION)
    assert not result.report.ok
    assert any("only one class" in e for e in result.report.errors)


def test_no_usable_rows_is_an_error():
    rows = [make_row(i, extraction_status="FAILED", **{n: "" for n in FEATURE_NAMES}) for i in range(4)]
    result = validate_manifest_frame(frame(rows), VERSION)
    assert not result.report.ok
    assert result.training_rows.empty


def test_mixed_extractor_versions_is_an_error():
    rows = [make_row(i) for i in range(5)] + [make_row(5, extractor_version="0.9.0")]
    result = validate_manifest_frame(frame(rows))
    assert any("different extractor versions" in e for e in result.report.errors)


def test_extractor_version_mismatch_is_an_error():
    result = validate_manifest_frame(good_frame(), "2.0.0+perm9+api9")
    assert any("Re-extract" in e for e in result.report.errors)


def test_synthetic_label_source_is_flagged():
    rows = [make_row(i, label_source="UNIT_TEST_SYNTHETIC fixtures") for i in range(6)]
    result = validate_manifest_frame(frame(rows), VERSION)
    assert result.report.stats["synthetic_training_rows"] == 6
    assert any("never be used as evidence" in w for w in result.report.warnings)


def test_small_dataset_and_imbalance_warn():
    rows = [make_row(i, label="MALWARE", sha256=sha(f"m{i}")) for i in range(9)] + [make_row(100, label="BENIGN")]
    result = validate_manifest_frame(frame(rows), VERSION)
    assert result.report.stats["class_imbalance_ratio"] == 9.0
    assert any("imbalance" in w.lower() for w in result.report.warnings)
    assert any("too noisy" in w for w in result.report.warnings)


# ---------------- files and helpers ----------------
def test_validate_manifest_file_roundtrip(tmp_path):
    path = tmp_path / "manifest.csv"
    good_frame().to_csv(path, index=False)
    result = validate_manifest_file(path, VERSION)
    assert result.report.ok
    assert len(result.training_rows) == 6


def test_validate_manifest_file_missing(tmp_path):
    result = validate_manifest_file(tmp_path / "missing.csv")
    assert not result.report.ok and "not found" in result.report.errors[0]


def test_validate_manifest_file_empty_csv(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    result = validate_manifest_file(path)
    assert not result.report.ok


@pytest.mark.parametrize(
    "value,expected",
    [(5, 5), ("7", 7), (" 7 ", 7), ("5.0", 5), (5.0, 5), (0, 0), ("0", 0),
     (-1, None), ("-1", None), (2.5, None), ("2.5", None), ("", None), (None, None),
     (float("nan"), None), (float("inf"), None), (True, None), ("abc", None), ("1e3", None)],
)
def test_parse_count(value, expected):
    assert parse_count(value) == expected
