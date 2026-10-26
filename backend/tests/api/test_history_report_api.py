"""GET /scans and GET /reports/{id}: history, pagination, per-user isolation, report content."""

from __future__ import annotations

import json

import pytest
from sqlalchemy import select

from app.database.models import Report, Scan
from tests.api.conftest import FakeModelService, auth_headers, upload


@pytest.fixture
def two_users(client, valid_apk_bytes, apk_bytes):
    a, b = auth_headers(client, "alice@example.com"), auth_headers(client, "bob@example.com")
    a_ids = [upload(client, a, valid_apk_bytes, "a1.apk").json()["id"],
             upload(client, a, apk_bytes(permissions=[], method_refs=[]), "a2.apk").json()["id"]]
    b_ids = [upload(client, b, valid_apk_bytes, "b1.apk").json()["id"]]
    return a, b, a_ids, b_ids


# ---------------------------------------------------------------- history
def test_history_is_empty_for_a_new_user(client, alice):
    assert client.get("/scans", headers=alice).json() == {"items": [], "total": 0, "limit": 50, "offset": 0}


def test_history_lists_only_my_scans_newest_first(client, two_users):
    a, b, a_ids, b_ids = two_users
    mine = client.get("/scans", headers=a).json()
    assert [i["id"] for i in mine["items"]] == sorted(a_ids, reverse=True) and mine["total"] == 2
    theirs = client.get("/scans", headers=b).json()
    assert [i["id"] for i in theirs["items"]] == b_ids and theirs["total"] == 1
    assert {i["apk_name"] for i in mine["items"]} == {"a1.apk", "a2.apk"}


def test_history_items_have_summary_fields_only(client, two_users):
    a, *_ = two_users
    item = client.get("/scans", headers=a).json()["items"][0]
    assert set(item) == {"id", "apk_name", "package_name", "sha256", "scan_date", "scan_status",
                         "classification", "risk_score", "risk_level"}


def test_history_pagination(client, alice, valid_apk_bytes):
    ids = [upload(client, alice, valid_apk_bytes).json()["id"] for _ in range(5)]
    page1 = client.get("/scans", headers=alice, params={"limit": 2}).json()
    page2 = client.get("/scans", headers=alice, params={"limit": 2, "offset": 2}).json()
    page3 = client.get("/scans", headers=alice, params={"limit": 2, "offset": 4}).json()
    got = [i["id"] for p in (page1, page2, page3) for i in p["items"]]
    assert got == sorted(ids, reverse=True) and page1["total"] == 5 and len(page3["items"]) == 1


@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 101}, {"offset": -1}, {"limit": "x"}])
def test_history_rejects_bad_paging(client, alice, params):
    assert client.get("/scans", headers=alice, params=params).status_code == 422


def test_sql_injection_in_paging_params_is_just_a_422(client, alice):
    assert client.get("/scans", headers=alice, params={"limit": "1; DROP TABLE scans"}).status_code == 422
    assert client.get("/scans", headers=alice).status_code == 200


# ---------------------------------------------------------------- reports
def test_owner_can_read_the_report(client, two_users):
    a, _, a_ids, _ = two_users
    r = client.get(f"/reports/{a_ids[0]}", headers=a)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["scan"]["id"] == a_ids[0] and body["scan"]["scan_status"] == "COMPLETED"
    assert body["explanation_source"] == "rule_based" and body["explanation"]
    assert body["findings"] and {f["label"] for f in body["findings"]} <= {"OBSERVED", "ML", "HEURISTIC"}
    assert body["limitations"] and body["recommendations"]
    assert set(body["evidence_labels"]) == {"OBSERVED", "ML", "HEURISTIC"}


def test_report_says_it_is_not_certainty_and_makes_no_accuracy_claim(client, two_users):
    a, _, a_ids, _ = two_users
    text = client.get(f"/reports/{a_ids[0]}", headers=a).text.lower()
    assert "accuracy" not in text and "100%" not in text and "guarantee" not in text
    assert "static" in text


def test_another_users_report_is_404_identical_to_a_missing_one(client, two_users):
    a, b, a_ids, b_ids = two_users
    foreign = client.get(f"/reports/{a_ids[0]}", headers=b)
    missing = client.get("/reports/999999", headers=b)
    assert foreign.status_code == missing.status_code == 404
    assert foreign.json() == missing.json()
    assert client.get(f"/reports/{b_ids[0]}", headers=a).status_code == 404


def test_report_requires_authentication(client, two_users):
    _, _, a_ids, _ = two_users
    assert client.get(f"/reports/{a_ids[0]}").status_code == 401
    assert client.get(f"/reports/{a_ids[0]}", headers={"Authorization": "Bearer nope"}).status_code == 401


@pytest.mark.parametrize("bad", ["abc", "0", "-1", "1.5", "99999999999999999999999"])
def test_report_id_must_be_a_positive_integer(client, alice, bad):
    assert client.get(f"/reports/{bad}", headers=alice).status_code == 422


def test_report_body_has_no_internal_fields(client, two_users, settings):
    a, _, a_ids, _ = two_users
    text = client.get(f"/reports/{a_ids[0]}", headers=a).text
    for needle in ("result_json", "password_hash", "user_id", "model_sha256", "feature_schema_hash", str(settings.model_dir), "Traceback"):
        assert needle not in text


def test_failed_scan_still_has_a_report_with_limitations(client, alice, valid_apk_bytes):
    sid = upload(client, alice, valid_apk_bytes[: len(valid_apk_bytes) * 9 // 10]).json()["id"]
    body = client.get(f"/reports/{sid}", headers=alice).json()
    assert body["scan"]["scan_status"] == "EXTRACTION_FAILED" and body["scan"]["prediction"] is None
    assert body["scan"]["classification"] == "NEEDS REVIEW"
    assert body["limitations"]


def test_report_without_a_model_does_not_invent_ml_findings(make_client, real_model_service_factory, valid_apk_bytes):
    c = make_client(model_service=real_model_service_factory())
    h = auth_headers(c)
    sid = upload(c, h, valid_apk_bytes).json()["id"]
    body = c.get(f"/reports/{sid}", headers=h).json()
    assert not any("score of" in f["text"] for f in body["findings"] if f["label"] == "ML")
    assert any("No model prediction" in f["text"] for f in body["findings"])
    assert body["scan"]["prediction"] is None


def test_report_is_stored_one_to_one_with_the_scan(client, two_users):
    with client.app.state.session_factory() as db:
        assert len(db.scalars(select(Scan)).all()) == len(db.scalars(select(Report)).all()) == 3
        for report in db.scalars(select(Report)):
            assert json.loads(report.findings) is not None
