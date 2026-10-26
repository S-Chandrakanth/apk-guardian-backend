"""POST /scan: workflow, invalid uploads, corrupt APKs, model problems, reuse, failures."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from sqlalchemy import select, update

from app.database.models import Report, Scan
from app.ml import feature_lists
from app.ml.schema import FEATURE_NAMES
from app.services.apk_features import ExtractorEnvironmentError
from tests.api.conftest import (
    FakeModelService,
    auth_headers,
    in_process_extract,
    make_synthetic_model_dir,
    upload,
    uploads_left,
)


def scans_in_db(client) -> list[Scan]:
    with client.app.state.session_factory() as db:
        return list(db.scalars(select(Scan).order_by(Scan.id)))


# ---------------------------------------------------------------- authentication
def test_anonymous_upload_is_refused_before_the_body_is_read(client, valid_apk_bytes):
    r = client.post("/scan", files={"file": ("a.apk", valid_apk_bytes)})
    assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
    assert scans_in_db(client) == []


def test_upload_with_an_invalid_token_is_401(client, valid_apk_bytes):
    r = upload(client, {"Authorization": "Bearer garbage.token.value"}, valid_apk_bytes)
    assert r.status_code == 401
    assert scans_in_db(client) == []


# ---------------------------------------------------------------- happy path
def test_scan_success_returns_features_hash_prediction_and_risk(client, alice, valid_apk_bytes, settings, model_service):
    import hashlib

    r = upload(client, alice, valid_apk_bytes, "My App.apk")
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["scan_status"] == "COMPLETED"
    assert body["sha256"] == hashlib.sha256(valid_apk_bytes).hexdigest()
    assert body["file_size"] == len(valid_apk_bytes)
    assert body["apk_name"] == "My App.apk"
    assert body["extraction"]["status"] == "OK" and body["extraction"]["error_code"] is None
    assert set(body["features"]) == set(FEATURE_NAMES)
    assert all(isinstance(v, int) for v in body["features"].values())
    assert body["features"]["permissions_count"] == 3
    assert body["prediction"]["malicious_probability"] == pytest.approx(0.9)
    assert body["prediction"]["benign_probability"] == pytest.approx(0.1)
    assert body["model"]["available"] is True and body["model"]["model_id"] == "a" * 12
    assert body["classification"] in {"BENIGN", "SUSPICIOUS", "MALICIOUS", "NEEDS REVIEW"}
    assert 0 <= body["risk"]["score"] <= 100 and body["risk"]["level"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert body["risk"]["basis"] == "ML_AND_HEURISTIC"
    assert body["reused_previous_result"] is False
    assert "never executed" in body["notice"]
    assert uploads_left(settings) == []  # temp file removed
    assert model_service.fake.calls == 1
    # the model received exactly the schema's features
    assert list(model_service.fake.seen[0]) == list(FEATURE_NAMES)


def test_scan_is_stored_with_a_report(client, alice, valid_apk_bytes):
    scan_id = upload(client, alice, valid_apk_bytes).json()["id"]
    rows = scans_in_db(client)
    assert [s.id for s in rows] == [scan_id]
    assert rows[0].extractor_version == feature_lists.extractor_version()
    assert rows[0].model_sha256 == "a" * 64
    with client.app.state.session_factory() as db:
        assert db.scalar(select(Report).where(Report.scan_id == scan_id)) is not None


def test_response_does_not_leak_paths_or_internals(client, alice, valid_apk_bytes, settings):
    text = upload(client, alice, valid_apk_bytes).text
    for needle in (str(settings.upload_tmp_dir), "Traceback", "scan_20", "tmp", ".joblib", "result_json", "password", "secret"):
        assert needle not in text


def test_high_probability_with_supporting_evidence_is_flagged(client, valid_apk_bytes, make_client):
    c = make_client(model_service=FakeModelService(0.95))
    r = upload(c, auth_headers(c), valid_apk_bytes).json()
    assert r["classification"] in {"MALICIOUS", "SUSPICIOUS"}
    assert r["risk"]["score"] >= 50


def test_low_probability_is_not_flagged(make_client, apk_bytes):
    c = make_client(model_service=FakeModelService(0.02))
    r = upload(c, auth_headers(c), apk_bytes(permissions=["android.permission.INTERNET"], method_refs=[])).json()
    assert r["classification"] == "BENIGN" and r["risk"]["level"] == "LOW"


def test_uncertain_probability_needs_review(make_client, valid_apk_bytes):
    c = make_client(model_service=FakeModelService(0.5))
    assert upload(c, auth_headers(c), valid_apk_bytes).json()["classification"] == "NEEDS REVIEW"


# ---------------------------------------------------------------- invalid uploads
@pytest.mark.parametrize("name", ["app.txt", "app.apk.exe", "app", "app.zip", ".apk.sh", "app.APK.bak"])
def test_wrong_extension_is_400(client, alice, valid_apk_bytes, name):
    assert upload(client, alice, valid_apk_bytes, name).status_code == 400
    assert scans_in_db(client) == []


def test_uppercase_apk_extension_is_accepted(client, alice, valid_apk_bytes):
    assert upload(client, alice, valid_apk_bytes, "APP.APK").status_code == 201


def test_text_file_renamed_to_apk_is_400_and_not_stored(client, alice, settings):
    r = upload(client, alice, b"this is plainly not an archive. " * 100)
    assert r.status_code == 400
    assert scans_in_db(client) == [] and uploads_left(settings) == []


@pytest.mark.parametrize("data", [b"", b"PK\x03\x04", b"x" * 100])
def test_empty_or_tiny_files_are_400(client, alice, data, settings):
    assert upload(client, alice, data).status_code == 400
    assert uploads_left(settings) == []


def test_oversize_upload_is_413_by_content_length(client, alice, settings):
    big = b"PK\x03\x04" + b"0" * (2 * 1024 * 1024)
    assert upload(client, alice, big).status_code == 413
    assert scans_in_db(client) == [] and uploads_left(settings) == []


def test_oversize_upload_within_multipart_allowance_is_still_413(client, alice, settings):
    just_over = b"PK\x03\x04" + b"0" * (settings.max_apk_size_bytes + 1000)
    r = upload(client, alice, just_over)
    assert r.status_code == 413
    assert scans_in_db(client) == [] and uploads_left(settings) == []


def test_request_without_a_file_is_422(client, alice):
    assert client.post("/scan", headers=alice, data={"x": "y"}).status_code == 422


@pytest.mark.parametrize("name", [
    "../../etc/passwd.apk",
    "..\\..\\windows\\evil.apk",
    "/abs/path/x.apk",
    "C:\\Users\\me\\x.apk",
    "x\r\ny.apk",
    "<script>alert(1)</script>.apk",
    ("a" * 400) + ".apk",
])
def test_hostile_filenames_are_sanitised_and_never_used_on_disk(client, alice, valid_apk_bytes, name, settings, tmp_path):
    r = upload(client, alice, valid_apk_bytes, name)
    assert r.status_code == 201, r.text
    shown = r.json()["apk_name"]
    assert "/" not in shown and "\\" not in shown and ".." not in shown and "\r" not in shown and "\n" not in shown
    assert len(shown) <= 255 and shown.lower().endswith(".apk")
    assert uploads_left(settings) == []
    assert not (tmp_path.parent / "etc").exists()


# ---------------------------------------------------------------- APK-level problems
def test_truncated_apk_is_stored_as_extraction_failed_with_safe_message(client, alice, valid_apk_bytes, settings):
    r = upload(client, alice, valid_apk_bytes[: len(valid_apk_bytes) * 9 // 10])
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["scan_status"] == "EXTRACTION_FAILED"
    assert body["extraction"]["status"] == "FAILED" and body["extraction"]["error_code"]
    assert body["prediction"] is None and all(v is None for v in body["features"].values())  # never zero-filled
    assert body["classification"] == "NEEDS REVIEW"
    assert "Traceback" not in r.text and "zipfile" not in r.text.lower()
    assert uploads_left(settings) == []


def test_zip_with_garbage_inside_is_failed_not_a_server_error(client, alice, settings):
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("readme.txt", "hello " * 200)
    r = upload(client, alice, buf.getvalue())
    assert r.status_code == 201
    assert r.json()["scan_status"] == "EXTRACTION_FAILED"
    assert r.json()["extraction"]["error_code"] == "MISSING_MANIFEST"


def test_apk_without_dex_is_partial_and_gets_no_prediction(client, alice, apk_bytes, model_service):
    r = upload(client, alice, apk_bytes(include_dex=False))
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["scan_status"] == "EXTRACTION_PARTIAL"
    assert body["extraction"]["error_code"] == "MISSING_DEX"
    assert body["prediction"] is None and model_service.fake.calls == 0
    assert body["classification"] == "NEEDS REVIEW"
    assert body["features"]["permissions_count"] is not None  # manifest features are real
    assert body["features"]["api_calls_count"] is None  # code features are unknown, not zero
    assert body["risk"]["basis"] == "HEURISTIC_PARTIAL"


def test_extractor_timeout_is_recorded_without_internal_text(make_client, valid_apk_bytes, settings):
    def timed_out(path, limits=None, timeout_seconds=0):
        return SimpleNamespace(
            status="FAILED", features={n: None for n in FEATURE_NAMES}, package_name=None,
            error_code="TIMEOUT", error_message="internal detail /srv/x", indicators={},
            extractor_version=feature_lists.extractor_version(),
        )

    c = make_client(model_service=FakeModelService(0.9), extract_fn=timed_out)
    r = upload(c, auth_headers(c), valid_apk_bytes)
    assert r.status_code == 201
    body = r.json()
    assert body["scan_status"] == "EXTRACTION_FAILED" and body["extraction"]["error_code"] == "TIMEOUT"
    assert "time limit" in body["extraction"]["message"] and "/srv/x" not in r.text
    assert "/srv/x" not in json.dumps([s.result_json for s in scans_in_db(c)])


def test_analysis_engine_unavailable_is_503(make_client, valid_apk_bytes, settings):
    def broken(path, limits=None, timeout_seconds=0):
        raise ExtractorEnvironmentError("androguard is not installed (internal detail)")

    c = make_client(model_service=FakeModelService(0.9), extract_fn=broken)
    r = upload(c, auth_headers(c), valid_apk_bytes)
    assert r.status_code == 503 and "internal detail" not in r.text
    assert uploads_left(settings) == [] and scans_in_db(c) == []


def test_unexpected_extractor_exception_is_a_generic_500_and_cleans_up(make_client, valid_apk_bytes, settings):
    def explode(path, limits=None, timeout_seconds=0):
        raise RuntimeError("secret internal detail /opt/app/file.py")

    c = make_client(model_service=FakeModelService(0.9), extract_fn=explode, raise_exceptions=False)
    r = upload(c, auth_headers(c), valid_apk_bytes)
    assert r.status_code == 500
    assert "secret internal detail" not in r.text and "/opt/app" not in r.text and "Traceback" not in r.text
    assert "request_id" in r.json()
    assert uploads_left(settings) == [] and scans_in_db(c) == []


def test_prediction_failure_is_recorded_not_faked(make_client, valid_apk_bytes):
    ms = FakeModelService(0.9)
    ms.fake.predict = lambda features: (_ for _ in ()).throw(ValueError("boom internal"))
    c = make_client(model_service=ms)
    r = upload(c, auth_headers(c), valid_apk_bytes)
    assert r.status_code == 201
    body = r.json()
    assert body["scan_status"] == "PREDICTION_FAILED" and body["prediction"] is None
    assert body["classification"] == "NEEDS REVIEW" and "boom internal" not in r.text


def test_server_busy_is_503_with_retry_after(client, alice, valid_apk_bytes, settings):
    sem = client.app.state.scan_semaphore
    for _ in range(settings.max_concurrent_scans):
        assert sem.acquire(blocking=False)
    try:
        r = upload(client, alice, valid_apk_bytes)
        assert r.status_code == 503 and r.headers["retry-after"]
        assert scans_in_db(client) == [] and uploads_left(settings) == []
    finally:
        for _ in range(settings.max_concurrent_scans):
            sem.release()
    assert upload(client, alice, valid_apk_bytes).status_code == 201  # slots were returned


def test_semaphore_is_released_after_failures(client, alice, settings):
    for _ in range(settings.max_concurrent_scans + 2):
        assert upload(client, alice, b"x" * 2000).status_code == 400
    assert client.app.state.scan_semaphore.acquire(blocking=False)
    client.app.state.scan_semaphore.release()


# ---------------------------------------------------------------- model unavailable (real ModelService)
def test_no_model_file_gives_an_explicit_status_and_no_prediction(make_client, real_model_service_factory, valid_apk_bytes):
    c = make_client(model_service=real_model_service_factory())
    r = upload(c, auth_headers(c), valid_apk_bytes)
    assert r.status_code == 201
    body = r.json()
    assert body["scan_status"] == "MODEL_UNAVAILABLE"
    assert body["model"]["status"] == "NOT_FOUND" and body["model"]["available"] is False
    assert body["prediction"] is None and body["classification"] == "NEEDS REVIEW"
    assert body["extraction"]["status"] == "OK" and body["features"]["permissions_count"] == 3  # extraction still real
    assert body["risk"]["basis"] == "HEURISTIC_ONLY"


def test_synthetic_model_is_blocked_by_default(make_client, real_model_service_factory, valid_apk_bytes, settings):
    make_synthetic_model_dir(settings.model_dir, synthetic=True)
    c = make_client(model_service=real_model_service_factory())
    body = upload(c, auth_headers(c), valid_apk_bytes).json()
    assert body["scan_status"] == "MODEL_UNAVAILABLE" and body["model"]["status"] == "SYNTHETIC_BLOCKED"
    assert body["prediction"] is None


def test_synthetic_model_can_be_allowed_explicitly_and_is_labelled(make_client, real_model_service_factory, valid_apk_bytes, settings):
    make_synthetic_model_dir(settings.model_dir, synthetic=True)
    c = make_client(model_service=real_model_service_factory(allow_synthetic=True))
    body = upload(c, auth_headers(c), valid_apk_bytes).json()
    assert body["scan_status"] == "COMPLETED"
    assert body["model"]["trained_on_synthetic_test_data"] is True
    assert 0.0 <= body["prediction"]["malicious_probability"] <= 1.0


def test_tampered_model_file_is_rejected(make_client, real_model_service_factory, valid_apk_bytes, settings):
    make_synthetic_model_dir(settings.model_dir, synthetic=False)
    model_file = next(settings.model_dir.glob("*.joblib"))
    model_file.write_bytes(model_file.read_bytes() + b"tampered")
    c = make_client(model_service=real_model_service_factory())
    body = upload(c, auth_headers(c), valid_apk_bytes).json()
    assert body["scan_status"] == "MODEL_UNAVAILABLE" and body["model"]["status"] in {"INCOMPATIBLE", "ERROR"}
    assert body["prediction"] is None


def test_schema_mismatch_model_is_incompatible(make_client, real_model_service_factory, valid_apk_bytes, settings):
    schema = make_synthetic_model_dir(settings.model_dir, synthetic=False)
    schema["feature_names"] = list(reversed(schema["feature_names"]))
    (settings.model_dir / "feature_schema.json").write_text(json.dumps(schema))
    c = make_client(model_service=real_model_service_factory())
    body = upload(c, auth_headers(c), valid_apk_bytes).json()
    assert body["model"]["status"] == "INCOMPATIBLE" and body["prediction"] is None


def test_model_installed_later_is_picked_up_without_restart(make_client, real_model_service_factory, valid_apk_bytes, settings):
    c = make_client(model_service=real_model_service_factory(allow_synthetic=True))
    headers = auth_headers(c)
    assert upload(c, headers, valid_apk_bytes).json()["scan_status"] == "MODEL_UNAVAILABLE"
    make_synthetic_model_dir(settings.model_dir, synthetic=True)
    assert upload(c, headers, valid_apk_bytes).json()["scan_status"] == "COMPLETED"


# ---------------------------------------------------------------- reuse of previous results
def counting_extract():
    calls = []

    def _extract(path, limits=None, timeout_seconds=0):
        calls.append(path)
        return in_process_extract(path, limits)

    return _extract, calls


def test_same_user_same_file_same_model_reuses_the_result(make_client, valid_apk_bytes):
    extract, calls = counting_extract()
    ms = FakeModelService(0.9)
    c = make_client(model_service=ms, extract_fn=extract)
    h = auth_headers(c)
    first = upload(c, h, valid_apk_bytes).json()
    second = upload(c, h, valid_apk_bytes).json()
    assert first["reused_previous_result"] is False and second["reused_previous_result"] is True
    assert second["id"] != first["id"]
    assert len(calls) == 1 and ms.fake.calls == 1
    assert second["features"] == first["features"] and second["prediction"] == first["prediction"]
    assert second["risk"]["score"] == first["risk"]["score"]


def test_changed_model_is_not_reused(make_client, valid_apk_bytes):
    extract, calls = counting_extract()
    ms = FakeModelService(0.9)
    c = make_client(model_service=ms, extract_fn=extract)
    h = auth_headers(c)
    upload(c, h, valid_apk_bytes)
    ms.sha = "b" * 64
    second = upload(c, h, valid_apk_bytes).json()
    assert second["reused_previous_result"] is False and len(calls) == 2 and ms.fake.calls == 2


def test_other_users_scans_are_never_reused(make_client, valid_apk_bytes):
    extract, calls = counting_extract()
    c = make_client(model_service=FakeModelService(0.9), extract_fn=extract)
    a, b = auth_headers(c, "alice@example.com"), auth_headers(c, "bob@example.com")
    upload(c, a, valid_apk_bytes)
    second = upload(c, b, valid_apk_bytes).json()
    assert second["reused_previous_result"] is False and len(calls) == 2


@pytest.mark.parametrize("column,value", [
    ("extractor_version", "0.0.0-old"),
    ("feature_schema_hash", "f" * 64),
    ("scan_status", "PREDICTION_FAILED"),
    ("model_sha256", None),
])
def test_incompatible_stored_rows_are_not_reused(make_client, valid_apk_bytes, column, value):
    extract, calls = counting_extract()
    c = make_client(model_service=FakeModelService(0.9), extract_fn=extract)
    h = auth_headers(c)
    upload(c, h, valid_apk_bytes)
    with c.app.state.session_factory() as db:
        db.execute(update(Scan).values({column: value}))
        db.commit()
    second = upload(c, h, valid_apk_bytes).json()
    assert second["reused_previous_result"] is False and len(calls) == 2


def test_no_reuse_when_no_model_is_available(make_client, real_model_service_factory, valid_apk_bytes):
    extract, calls = counting_extract()
    c = make_client(model_service=real_model_service_factory(), extract_fn=extract)
    h = auth_headers(c)
    upload(c, h, valid_apk_bytes)
    assert upload(c, h, valid_apk_bytes).json()["reused_previous_result"] is False and len(calls) == 2


# ---------------------------------------------------------------- the real isolated worker process
def test_real_isolated_extractor_end_to_end(make_client, valid_apk_bytes, settings):
    c = make_client(model_service=FakeModelService(0.9), extract_fn=None)  # None = the production default
    r = upload(c, auth_headers(c), valid_apk_bytes)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["scan_status"] == "COMPLETED" and body["features"]["permissions_count"] == 3
    assert uploads_left(settings) == []


def test_real_isolated_extractor_handles_a_corrupt_apk(make_client, valid_apk_bytes):
    c = make_client(model_service=FakeModelService(0.9), extract_fn=None)
    r = upload(c, auth_headers(c), valid_apk_bytes[: len(valid_apk_bytes) * 9 // 10])
    assert r.status_code == 201 and r.json()["scan_status"] == "EXTRACTION_FAILED"
