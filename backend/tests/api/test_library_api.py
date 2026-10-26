"""Tests for the Default APK Library API endpoints:
GET /library
POST /library/import
POST /library/{sample_id}/scan
DELETE /library/{sample_id}
"""

from __future__ import annotations

import hashlib
import io
import pytest
from sqlalchemy import select

from app.database.models import LibrarySample, Scan
from tests.api.conftest import (
    FakeModelService,
    auth_headers,
    upload,
)


def test_library_list_empty(client, alice):
    r = client.get("/library", headers=alice)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"] == 0
    assert body["samples"] == []


def test_library_import_apk(client, alice, valid_apk_bytes):
    expected_sha = hashlib.sha256(valid_apk_bytes).hexdigest()
    
    files = {"file": ("sample_test.apk", io.BytesIO(valid_apk_bytes), "application/vnd.android.package-archive")}
    data = {"label": "benign", "label_source": "user"}
    r = client.post("/library/import", headers=alice, files=files, data=data)
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["is_duplicate"] is False
    assert res["sample"]["filename"] == "sample_test.apk"
    assert res["sample"]["sha256"] == expected_sha
    assert res["sample"]["ground_truth_label"] == "benign"
    assert res["sample"]["label_source"] == "user"
    assert res["sample"]["last_scan_status"] == "NOT_SCANNED"

    # Check listing includes it
    list_r = client.get("/library", headers=alice)
    assert list_r.status_code == 200
    list_body = list_r.json()
    assert list_body["total"] == 1
    assert list_body["samples"][0]["sha256"] == expected_sha


def test_library_import_duplicate_prevention(client, alice, valid_apk_bytes):
    files1 = {"file": ("sample1.apk", io.BytesIO(valid_apk_bytes), "application/vnd.android.package-archive")}
    r1 = client.post("/library/import", headers=alice, files=files1)
    assert r1.status_code == 200
    assert r1.json()["is_duplicate"] is False

    # Import identical APK under a different filename
    files2 = {"file": ("sample2.apk", io.BytesIO(valid_apk_bytes), "application/vnd.android.package-archive")}
    r2 = client.post("/library/import", headers=alice, files=files2)
    assert r2.status_code == 200
    assert r2.json()["is_duplicate"] is True
    assert "already exists" in r2.json()["message"]

    # Library total should still be 1
    list_r = client.get("/library", headers=alice)
    assert list_r.json()["total"] == 1


def test_library_scan_sample(client, alice, valid_apk_bytes, model_service):
    # Import sample
    files = {"file": ("scan_me.apk", io.BytesIO(valid_apk_bytes), "application/vnd.android.package-archive")}
    r_imp = client.post("/library/import", headers=alice, files=files)
    sample_id = r_imp.json()["sample"]["id"]

    # Trigger scan
    r_scan = client.post(f"/library/{sample_id}/scan", headers=alice)
    assert r_scan.status_code == 200, r_scan.text
    scan_body = r_scan.json()
    assert scan_body["scan_status"] == "COMPLETED"
    assert scan_body["sha256"] == hashlib.sha256(valid_apk_bytes).hexdigest()
    assert scan_body["prediction"]["malicious_probability"] is not None

    # Check sample metadata updated
    list_r = client.get("/library", headers=alice)
    sample = list_r.json()["samples"][0]
    assert sample["last_scan_status"] == "COMPLETED"
    assert sample["last_prediction"] is not None


def test_library_delete_sample_keeps_scan_history(client, alice, valid_apk_bytes, model_service):
    # Import and scan
    files = {"file": ("delete_me.apk", io.BytesIO(valid_apk_bytes), "application/vnd.android.package-archive")}
    r_imp = client.post("/library/import", headers=alice, files=files)
    sample_id = r_imp.json()["sample"]["id"]
    client.post(f"/library/{sample_id}/scan", headers=alice)

    # Verify scan exists in DB
    with client.app.state.session_factory() as db:
        scans = list(db.scalars(select(Scan)))
        assert len(scans) == 1

    # Delete sample from library
    r_del = client.delete(f"/library/{sample_id}", headers=alice)
    assert r_del.status_code == 200
    assert r_del.json()["deleted"] is True

    # Library should be empty
    list_r = client.get("/library", headers=alice)
    assert list_r.json()["total"] == 0

    # History/Scan records must remain intact!
    with client.app.state.session_factory() as db:
        scans = list(db.scalars(select(Scan)))
        assert len(scans) == 1


def test_library_download_sample(client, alice, valid_apk_bytes):
    files = {"file": ("dl_test.apk", io.BytesIO(valid_apk_bytes), "application/vnd.android.package-archive")}
    r_imp = client.post("/library/import", headers=alice, files=files)
    sample_id = r_imp.json()["sample"]["id"]

    r_dl = client.get(f"/library/{sample_id}/download", headers=alice)
    assert r_dl.status_code == 200
    assert r_dl.content == valid_apk_bytes

