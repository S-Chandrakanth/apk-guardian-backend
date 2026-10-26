"""Capture REAL backend responses into frontend/test/fixtures/ for the Flutter parser tests.

Run from the backend folder:   python tools\\capture_frontend_fixtures.py

Responses come from the actual FastAPI app (temporary database, benign synthetic test APKs from
tests/fixtures/apk_builder.py). The default fixtures are produced with NO model installed, which is
the real current state (MODEL_UNAVAILABLE, prediction null). One extra fixture uses a FAKE model
only so the Dart parser can be tested on a response that contains a prediction; its file name says so.
No data here is evidence of detection quality.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from fastapi.testclient import TestClient  # noqa: E402

from app.config import Settings  # noqa: E402
from app.database.session import init_db  # noqa: E402
from app.main import create_app  # noqa: E402
from app.ml import feature_lists  # noqa: E402
from app.ml.predictor import Prediction  # noqa: E402
from app.services.apk_features import extract_features  # noqa: E402
from app.services.model_service import AVAILABLE, ModelService, ModelStatus  # noqa: E402
from tests.fixtures.apk_builder import build_apk  # noqa: E402

OUT = BACKEND.parent / "frontend" / "test" / "fixtures"
PASSWORD = "a long passphrase 42"


class _FakePredictor:
    def predict(self, features):
        return Prediction(0.9, 0.1, "MALWARE", "fake.joblib", feature_lists.extractor_version(), False)


class _FakeModelService:
    def status(self):
        return ModelStatus(AVAILABLE, "A compatible trained model is loaded.", feature_lists.extractor_version(), "a" * 64, False)

    def predictor(self):
        return _FakePredictor()


def _extract(path, limits=None, timeout_seconds=0):
    return extract_features(path, limits)


def _save(name: str, data) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote", name)


def _make_client(tmp: Path, model_service=None) -> TestClient:
    settings = Settings(
        _env_file=None, secret_key="fixture-secret-" + "k" * 40,
        database_url=f"sqlite:///{(tmp / 'fx.db').as_posix()}", upload_tmp_dir=tmp / "up",
        model_dir=tmp / "models", max_apk_size_mb=1, min_apk_size_bytes=512, login_max_failures=3,
        scan_queue_wait_seconds=0,
    )
    app = create_app(settings, model_service=model_service, extract_fn=_extract)
    init_db(app.state.engine)
    return TestClient(app)


def main() -> int:
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        good = build_apk(
            tmp / "good.apk",
            permissions=["android.permission.INTERNET", "android.permission.SEND_SMS", "android.permission.READ_SMS"],
            method_refs=[("Landroid/telephony/SmsManager;", "sendTextMessage", "V", ("Ljava/lang/String;",)),
                         ("Landroid/app/Activity;", "onCreate", "V", ("Landroid/os/Bundle;",))],
        ).read_bytes()
        nodex = build_apk(tmp / "nodex.apk", include_dex=False).read_bytes()
        broken = good[: len(good) * 9 // 10]

        with _make_client(tmp) as c:  # no model installed: the real current state
            reg = c.post("/register", json={"name": "Fixture User", "email": "fixture@example.com", "password": PASSWORD})
            _save("register", reg.json())
            _save("error_409_duplicate", c.post("/register", json={"name": "F", "email": "fixture@example.com", "password": PASSWORD}).json())
            _save("error_422_validation", c.post("/register", json={"name": "F", "email": "bad", "password": "short"}).json())
            login = c.post("/login", json={"email": "fixture@example.com", "password": PASSWORD}).json()
            login["access_token"] = "FIXTURE-TOKEN-NOT-A-REAL-TOKEN"
            _save("login", login)
            _save("error_401_login", c.post("/login", json={"email": "fixture@example.com", "password": "wrong password"}).json())
            token = c.post("/login", json={"email": "fixture@example.com", "password": PASSWORD}).json()["access_token"]
            h = {"Authorization": f"Bearer {token}"}
            _save("me", c.get("/me", headers=h).json())
            _save("health_no_model", c.get("/health").json())
            _save("scans_summary_empty", c.get("/scans/summary", headers=h).json())
            _save("scans_empty", c.get("/scans", headers=h).json())

            def scan(name, data, filename):
                r = c.post("/scan", headers=h, files={"file": (filename, data)})
                assert r.status_code == 201, r.text
                _save(name, r.json())
                return r.json()

            ok = scan("scan_model_unavailable", good, "sample-app.apk")
            scan("scan_partial_no_dex", nodex, "no-dex.apk")
            scan("scan_failed_corrupt", broken, "damaged.apk")
            _save("report_model_unavailable", c.get(f"/reports/{ok['id']}", headers=h).json())
            _save("scans_list", c.get("/scans", headers=h).json())
            _save("scans_summary", c.get("/scans/summary", headers=h).json())
            _save("error_404_report", c.get("/reports/99999", headers=h).json())
            _save("error_400_not_apk", c.post("/scan", headers=h, files={"file": ("x.apk", b"not an archive " * 100)}).json())
            _save("error_401_missing_token", c.get("/scans").json())

        with _make_client(tmp / "fake", model_service=_FakeModelService()) as c2:
            c2.post("/register", json={"name": "Fixture User", "email": "fixture@example.com", "password": PASSWORD})
            tok = c2.post("/login", json={"email": "fixture@example.com", "password": PASSWORD}).json()["access_token"]
            h2 = {"Authorization": f"Bearer {tok}"}
            r = c2.post("/scan", headers=h2, files={"file": ("sample-app.apk", good)})
            body = r.json()
            body["_fixture_note"] = "FAKE MODEL used only to test the parser on a response with a prediction. Not a real result."
            _save("scan_completed__fake_model_parser_test_only", body)
            _save("health_fake_model__parser_test_only", c2.get("/health").json())
    return 0


if __name__ == "__main__":
    sys.exit(main())
