"""Backend additions made for the Flutter app: GET /scans/summary and the `indicators` field."""

from __future__ import annotations

import pytest

from tests.api.conftest import FakeModelService, auth_headers, upload

STATUSES = {"COMPLETED", "MODEL_UNAVAILABLE", "EXTRACTION_PARTIAL", "EXTRACTION_FAILED", "PREDICTION_FAILED"}


# ---------------------------------------------------------------- /scans/summary
def test_summary_requires_authentication(client):
    assert client.get("/scans/summary").status_code == 401


def test_summary_for_a_new_user_has_every_key_with_zero(client, alice):
    body = client.get("/scans/summary", headers=alice).json()
    assert body["total"] == 0 and body["last_scan_at"] is None
    assert set(body["by_status"]) == STATUSES and set(body["by_status"].values()) == {0}
    assert set(body["by_classification"]) == {"BENIGN", "SUSPICIOUS", "MALICIOUS", "NEEDS REVIEW"}
    assert set(body["by_risk_level"]) == {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


def test_summary_counts_come_from_real_scans(client, alice, valid_apk_bytes, apk_bytes):
    upload(client, alice, valid_apk_bytes, "a.apk")
    upload(client, alice, apk_bytes(include_dex=False), "b.apk")
    upload(client, alice, valid_apk_bytes[: len(valid_apk_bytes) * 9 // 10], "c.apk")
    body = client.get("/scans/summary", headers=alice).json()
    assert body["total"] == 3
    assert body["by_status"]["COMPLETED"] == 1
    assert body["by_status"]["EXTRACTION_PARTIAL"] == 1
    assert body["by_status"]["EXTRACTION_FAILED"] == 1
    assert sum(body["by_status"].values()) == 3
    assert body["last_scan_at"].endswith("Z") or "+" in body["last_scan_at"]
    assert sum(body["by_classification"].values()) == 3  # every scan has a classification
    assert sum(body["by_risk_level"].values()) == 2  # the failed scan has no score and is not counted


def test_summary_is_per_user(client, alice, bob, valid_apk_bytes):
    upload(client, alice, valid_apk_bytes)
    assert client.get("/scans/summary", headers=alice).json()["total"] == 1
    assert client.get("/scans/summary", headers=bob).json()["total"] == 0


def test_summary_matches_the_history_total(client, alice, valid_apk_bytes):
    for _ in range(3):
        upload(client, alice, valid_apk_bytes)
    assert client.get("/scans/summary", headers=alice).json()["total"] == client.get("/scans", headers=alice).json()["total"]


def test_summary_route_is_not_shadowed_by_the_report_route(client, alice):
    assert client.get("/scans/summary", headers=alice).status_code == 200
    assert client.get("/reports/summary", headers=alice).status_code == 422


# ---------------------------------------------------------------- indicators
def test_scan_exposes_the_names_behind_the_counts(client, alice, valid_apk_bytes):
    body = upload(client, alice, valid_apk_bytes).json()
    ind = body["indicators"]
    assert len(ind["permissions"]) == body["features"]["permissions_count"] == 3
    assert "android.permission.SEND_SMS" in ind["permissions"]
    assert set(ind["dangerous_permissions"]) <= set(ind["permissions"])
    assert len(ind["dangerous_permissions"]) == body["features"]["dangerous_permissions"]
    assert len(ind["activities"]) == body["features"]["activities_count"]
    assert len(ind["services"]) == body["features"]["services_count"]
    assert len(ind["receivers"]) == body["features"]["receivers_count"]
    assert len(ind["suspicious_apis"]) == body["features"]["suspicious_api_calls"]


def test_missing_dex_gives_null_api_names_not_an_empty_list(client, alice, apk_bytes):
    body = upload(client, alice, apk_bytes(include_dex=False)).json()
    assert body["indicators"]["suspicious_apis"] is None
    assert body["indicators"]["permissions"] is not None


def test_failed_extraction_has_no_indicators(client, alice, valid_apk_bytes):
    body = upload(client, alice, valid_apk_bytes[: len(valid_apk_bytes) * 9 // 10]).json()
    assert body["scan_status"] == "EXTRACTION_FAILED" and body["indicators"] is None


def test_indicators_are_also_in_the_report_and_survive_reuse(make_client, valid_apk_bytes):
    c = make_client(model_service=FakeModelService(0.9))
    h = auth_headers(c)
    first = upload(c, h, valid_apk_bytes).json()
    second = upload(c, h, valid_apk_bytes).json()
    assert second["reused_previous_result"] is True and second["indicators"] == first["indicators"]
    assert c.get(f"/reports/{first['id']}", headers=h).json()["scan"]["indicators"] == first["indicators"]


def test_indicator_lists_are_bounded(client, alice, valid_apk_bytes):
    from app.services.presenters import MAX_LISTED_NAMES, MAX_NAME_LENGTH, _names

    out = _names([("x" * 1000)] * (MAX_LISTED_NAMES + 50))
    assert len(out) == MAX_LISTED_NAMES and all(len(n) == MAX_NAME_LENGTH for n in out)
    assert _names("not a list") is None and _names(None) is None


@pytest.mark.parametrize("stored", [None, {}, {"activities": []}, "oops"])
def test_old_or_odd_stored_rows_give_null_indicators(stored):
    from app.services.presenters import _indicators

    assert _indicators(stored) is None
