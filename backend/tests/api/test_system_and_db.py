"""System endpoints, database constraints and failures, HTTP behaviour, small unit tests."""

from __future__ import annotations

import asyncio
import io

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from app.database.models import Report, Scan, User
from app.database.session import database_status, init_db, make_engine
from app.middleware import RequestGuardMiddleware
from app.services.upload import UploadRejected, remove_quietly, sanitize_filename, save_upload
from tests.api.conftest import FakeModelService, auth_headers, register, upload, uploads_left


# ---------------------------------------------------------------- system endpoints
def test_root(client):
    r = client.get("/")
    assert r.status_code == 200 and r.json()["status"] == "running" and r.json()["docs"] == "/docs"


def test_health_ok_with_a_model(client):
    r = client.get("/health")
    body = r.json()
    assert r.status_code == 200 and body["status"] == "ok" and body["database"] == "ok"
    assert body["analysis_engine"]["available"] is True and body["analysis_engine"]["androguard_version"]
    assert body["model"]["available"] is True


def test_health_degraded_without_a_model_still_200(make_client):
    r = make_client(model_service=FakeModelService(None)).get("/health")
    assert r.status_code == 200 and r.json()["status"] == "degraded" and r.json()["model"]["status"] == "NOT_FOUND"


def test_health_unavailable_when_database_is_not_initialised(make_client):
    r = make_client(model_service=FakeModelService(0.9), init=False).get("/health")
    assert r.status_code == 503 and r.json()["status"] == "unavailable" and r.json()["database"] == "not_initialized"


def test_health_does_not_expose_secrets_or_paths(client, settings):
    body = client.get("/health").text
    assert settings.secret_key not in body and str(settings.model_dir) not in body and "sqlite" not in body.lower()


def test_openapi_and_docs_are_served_and_declare_bearer_auth(client):
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json").json()
    assert {"/", "/health", "/register", "/login", "/scan", "/scans", "/reports/{scan_id}"} <= set(schema["paths"])
    assert any(v.get("scheme") == "bearer" for v in schema["components"]["securitySchemes"].values())


# ---------------------------------------------------------------- HTTP behaviour
def test_security_headers_and_request_id(client):
    r = client.get("/")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["cache-control"] == "no-store" and r.headers["referrer-policy"] == "no-referrer"
    assert len(r.headers["x-request-id"]) == 16
    assert r.headers["x-request-id"] != client.get("/").headers["x-request-id"]


def test_cors_allows_the_configured_origin_only(client):
    ok = client.options("/scans", headers={"Origin": "http://localhost:8080", "Access-Control-Request-Method": "GET",
                                           "Access-Control-Request-Headers": "Authorization"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:8080"
    bad = client.options("/scans", headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "GET"})
    assert "access-control-allow-origin" not in bad.headers
    assert "access-control-allow-credentials" not in ok.headers


def test_unknown_route_is_404_and_wrong_method_405(client):
    assert client.get("/nope").status_code == 404
    assert client.get("/scan", headers={"Authorization": "Bearer x"}).status_code in (401, 405)
    assert client.delete("/scans").status_code == 405


# ---------------------------------------------------------------- database constraints
def test_unique_email_and_lowercase_constraints(client):
    with client.app.state.session_factory() as db:
        db.add(User(name="A", email="a@example.com", password_hash="x"))
        db.commit()
        db.add(User(name="B", email="a@example.com", password_hash="y"))
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()
        db.add(User(name="C", email="UPPER@example.com", password_hash="z"))
        with pytest.raises(IntegrityError):
            db.commit()


def test_scan_check_constraints_reject_bad_values(client, alice, valid_apk_bytes):
    upload(client, alice, valid_apk_bytes)
    with client.app.state.session_factory() as db:
        scan = db.scalar(select(Scan))
        scan.scan_status = "MADE_UP"
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()
        scan.risk_score = 101
        with pytest.raises(IntegrityError):
            db.commit()


def test_foreign_keys_are_enforced_and_deleting_a_user_cascades(client, alice, valid_apk_bytes):
    upload(client, alice, valid_apk_bytes)
    with client.app.state.session_factory() as db:
        assert db.execute(text("PRAGMA foreign_keys")).scalar() == 1
        db.add(Scan(user_id=9999, apk_name="x.apk", sha256="0" * 64, file_size=1, scan_status="COMPLETED",
                    extraction_status="OK", extractor_version="v", feature_schema_hash="h", risk_logic_version="r",
                    result_json="{}"))
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()
        db.execute(text("DELETE FROM users"))
        db.commit()
        assert db.scalar(select(Scan)) is None and db.scalar(select(Report)) is None


def test_init_db_is_repeatable_and_sets_schema_version(settings):
    engine = make_engine(settings)
    assert database_status(engine) == "not_initialized"
    init_db(engine)
    init_db(engine)
    assert database_status(engine) == "ok"
    with engine.connect() as conn:
        assert conn.execute(text("PRAGMA user_version")).scalar() == 1


def test_database_status_detects_a_schema_version_mismatch(settings):
    engine = make_engine(settings)
    init_db(engine)
    with engine.begin() as conn:
        conn.execute(text("PRAGMA user_version = 99"))
    assert database_status(engine) == "schema_mismatch"


# ---------------------------------------------------------------- database failures through the API
def test_uninitialised_database_gives_generic_503(make_client):
    c = make_client(model_service=FakeModelService(0.9), init=False, raise_exceptions=False)
    r = register(c)
    assert r.status_code == 503
    assert "no such table" not in r.text.lower() and "sql" not in r.text.lower() and "request_id" in r.json()


def test_operational_error_is_a_generic_503(make_client, monkeypatch):
    c = make_client(model_service=FakeModelService(0.9), raise_exceptions=False)

    def broken(self, *args, **kwargs):
        raise OperationalError("SELECT secret_column FROM users", {}, Exception("disk I/O error at /var/db"))

    monkeypatch.setattr(Session, "scalar", broken)
    r = c.post("/login", json={"email": "a@example.com", "password": "some password 123"})
    assert r.status_code == 503
    for needle in ("secret_column", "disk I/O", "/var/db", "SELECT", "Traceback"):
        assert needle not in r.text


def test_commit_failure_during_scan_is_503_leaves_no_rows_and_cleans_up(make_client, valid_apk_bytes, settings, monkeypatch):
    c = make_client(model_service=FakeModelService(0.9), raise_exceptions=False)
    headers = auth_headers(c)
    real_commit = Session.commit

    def failing_commit(self):
        if any(isinstance(obj, Scan) for obj in self.new):
            raise OperationalError("INSERT INTO scans ...", {}, Exception("database is locked"))
        return real_commit(self)

    monkeypatch.setattr(Session, "commit", failing_commit)
    r = upload(c, headers, valid_apk_bytes)
    assert r.status_code == 503 and "locked" not in r.text and "INSERT" not in r.text
    monkeypatch.undo()
    assert uploads_left(settings) == []
    with c.app.state.session_factory() as db:
        assert db.scalar(select(Scan)) is None and db.scalar(select(Report)) is None
    assert upload(c, headers, valid_apk_bytes).status_code == 201  # the service recovered


# ---------------------------------------------------------------- unit tests: upload and middleware
def test_save_upload_enforces_the_cap_and_removes_partial_files(tmp_path):
    with pytest.raises(UploadRejected) as exc:
        save_upload(io.BytesIO(b"PK\x03\x04" + b"0" * 5000), tmp_path, max_bytes=1000, min_bytes=10)
    assert exc.value.status_code == 413 and list(tmp_path.iterdir()) == []


def test_save_upload_hashes_what_it_stored(tmp_path):
    import hashlib

    data = b"PK\x03\x04" + b"abc" * 1000
    saved = save_upload(io.BytesIO(data), tmp_path, max_bytes=10_000, min_bytes=10)
    assert saved.size == len(data) and saved.sha256 == hashlib.sha256(data).hexdigest()
    assert saved.path.read_bytes() == data and saved.path.parent == tmp_path
    remove_quietly(saved.path)
    assert not saved.path.exists()
    remove_quietly(saved.path)  # removing twice is harmless


@pytest.mark.parametrize("raw,expected", [
    ("../../etc/passwd.apk", "passwd.apk"),
    ("..\\..\\evil.apk", "evil.apk"),
    ("", "upload.apk"),
    (None, "upload.apk"),
    ("some/dir/", "upload.apk"),
    ("my app (1).apk", "my app (1).apk"),
])
def test_sanitize_filename(raw, expected):
    assert sanitize_filename(raw) == expected


def test_stale_upload_cleanup_removes_only_old_scan_files(tmp_path):
    import os
    import time

    from app.services.upload import cleanup_stale_uploads

    old, fresh, other = tmp_path / "scan_old.apk", tmp_path / "scan_new.apk", tmp_path / "keep.txt"
    for f in (old, fresh, other):
        f.write_bytes(b"x")
    past = time.time() - 7200
    os.utime(old, (past, past))
    os.utime(other, (past, past))
    cleanup_stale_uploads(tmp_path)
    assert not old.exists() and fresh.exists() and other.exists()


def test_middleware_counts_streamed_bytes_when_there_is_no_content_length():
    """A chunked request that never declares its size is still stopped."""
    reached = []

    async def app(scope, receive, send):
        while True:
            message = await receive()
            if not message.get("more_body"):
                break
        reached.append(True)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    mw = RequestGuardMiddleware(app, upload_paths={"/scan"}, upload_limit=100, json_limit=100)
    chunks = [{"type": "http.request", "body": b"x" * 40, "more_body": True} for _ in range(10)]
    chunks.append({"type": "http.request", "body": b"", "more_body": False})
    sent: list[dict] = []

    async def receive():
        return chunks.pop(0)

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "method": "POST", "path": "/other", "headers": []}
    asyncio.run(mw(scope, receive, send))
    assert sent[0]["status"] == 413
    assert not any(m.get("body") == b"ok" for m in sent)
