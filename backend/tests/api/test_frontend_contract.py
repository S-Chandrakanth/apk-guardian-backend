"""Contract checks between the backend and the Flutter app.

1. Every success fixture the Dart tests use must validate against the real
   Pydantic response schemas (so fixtures cannot drift from the API).
2. Every JSON key the Dart models read must exist in some backend response schema
   (so a renamed backend field is caught here, without needing a Flutter SDK).

This does NOT prove the Dart code compiles; it only guards the data contract.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from app.schemas.auth import TokenOut, UserOut
from app.schemas.library import LibraryDeleteResponse, LibraryImportResponse, LibraryListResponse, LibrarySampleItem
from app.schemas.scans import ReportOut, ScanListOut, ScanOut, ScanSummaryOut

FRONTEND = Path(__file__).resolve().parents[3] / "frontend"
FIXTURES = FRONTEND / "test" / "fixtures"
MODELS = FRONTEND / "lib" / "models"

pytestmark = pytest.mark.skipif(not FIXTURES.is_dir(), reason="frontend/ folder not present")

SCHEMA_FOR_FIXTURE = {
    "register.json": UserOut,
    "me.json": UserOut,
    "login.json": TokenOut,
    "scan_model_unavailable.json": ScanOut,
    "scan_partial_no_dex.json": ScanOut,
    "scan_failed_corrupt.json": ScanOut,
    "scan_completed__fake_model_parser_test_only.json": ScanOut,
    "report_model_unavailable.json": ReportOut,
    "scans_list.json": ScanListOut,
    "scans_empty.json": ScanListOut,
    "scans_summary.json": ScanSummaryOut,
    "scans_summary_empty.json": ScanSummaryOut,
}


def _load(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", sorted(SCHEMA_FOR_FIXTURE))
def test_fixture_validates_against_backend_schema(name):
    SCHEMA_FOR_FIXTURE[name].model_validate(_load(name))


def test_every_fixture_is_valid_json():
    for path in FIXTURES.glob("*.json"):
        json.loads(path.read_text(encoding="utf-8"))


def test_login_fixture_holds_no_real_token():
    assert _load("login.json")["access_token"].startswith("FIXTURE-")


def test_fake_model_fixtures_are_labelled_parser_test_only():
    assert (FIXTURES / "scan_completed__fake_model_parser_test_only.json").exists()
    assert (FIXTURES / "health_fake_model__parser_test_only.json").exists()


def _schema_field_names() -> set[str]:
    names: set[str] = set()
    schemas = [
        UserOut, TokenOut, ScanOut, ReportOut, ScanListOut, ScanSummaryOut,
        LibraryListResponse, LibrarySampleItem, LibraryImportResponse, LibraryDeleteResponse,
    ]
    seen = set()

    def walk(model):
        if model in seen:
            return
        seen.add(model)
        names.update(model.model_fields)
        for field in model.model_fields.values():
            stack = [field.annotation]
            while stack:
                ann = stack.pop()
                if hasattr(ann, "model_fields"):
                    walk(ann)
                stack.extend(getattr(ann, "__args__", ()))

    for s in schemas:
        walk(s)
    return names


HELPER_KEY = re.compile(
    r"\b(?:asMap|reqString|optString|optInt|reqInt|optBool|reqBool|optDouble|reqDouble|"
    r"optStringList|intMap|optMap|optList|reqList)\(\s*\w+\s*,\s*'([A-Za-z_]+)'"
)
# Keys of the /health response are defined in app/main.py (not a Pydantic model).
HEALTH_ONLY = {
    "database", "analysis_engine", "androguard_version", "limits",
    "max_apk_size_mb", "analysis_timeout_seconds",
}
# Keys of the {"detail": ...} error body, read by the API client, not by models.
ERROR_KEYS = {"detail", "loc", "msg", "type", "request_id"}
# asMap(json, '<label>') on a top-level object uses the label only in error messages.
TOP_LEVEL_LABELS = {"health", "finding", "report", "token", "user", "library_sample", "library_list"}


def test_dart_models_only_read_keys_the_backend_sends():
    known = _schema_field_names() | HEALTH_ONLY | ERROR_KEYS | TOP_LEVEL_LABELS
    missing: dict[str, list[str]] = {}
    total = 0
    for path in sorted(MODELS.glob("*.dart")):
        keys = HELPER_KEY.findall(path.read_text(encoding="utf-8"))
        total += len(keys)
        bad = sorted({k for k in keys if k not in known})
        if bad:
            missing[path.name] = bad
    assert total > 40, "key extraction found suspiciously few keys; the regex may be stale"
    assert not missing, f"Dart reads keys no backend schema defines: {missing}"
