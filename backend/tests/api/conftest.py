"""Fixtures for the API tests.

Everything uses temporary directories and temporary SQLite files. Models are either fakes
or tiny classifiers trained on SYNTHETIC test numbers (never real data, never real malware).
APKs are the benign synthetic files from tests/fixtures/apk_builder.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.database.session import init_db
from app.main import create_app
from app.ml import feature_lists
from app.ml.predictor import Prediction
from app.security import passwords
from app.services.apk_features import extract_features
from app.services.model_service import AVAILABLE, ModelService, ModelStatus

PASSWORD = "correct horse battery 42"
SECRET = "test-secret-" + "k" * 40
FAKE_MODEL_SHA = "a" * 64


@pytest.fixture(autouse=True)
def _cheap_password_hashing(monkeypatch):
    """scrypt at production cost is deliberately slow; use a cheap setting for the test suite.
    tests/api/test_security.py checks the production parameters explicitly."""
    monkeypatch.setattr(passwords, "SCRYPT_N", 2 ** 10)
    monkeypatch.setattr(passwords, "SCRYPT_P", 1)


# ---------------------------------------------------------------- model fakes
@dataclass
class FakePredictor:
    malware_probability: float = 0.9
    calls: int = 0
    seen: list[dict[str, Any]] = field(default_factory=list)

    def predict(self, features: dict[str, Any]) -> Prediction:
        self.calls += 1
        self.seen.append(dict(features))
        p = self.malware_probability
        return Prediction(
            malware_probability=p,
            benign_probability=1 - p,
            raw_label="MALWARE" if p >= 0.5 else "BENIGN",
            model_file="fake.joblib",
            extractor_version=feature_lists.extractor_version(),
            trained_on_synthetic_test_data=False,
        )


class FakeModelService:
    """Stands in for ModelService with a controllable predictor and status."""

    def __init__(self, probability: Optional[float] = 0.9, sha: str = FAKE_MODEL_SHA) -> None:
        self.sha = sha
        self.fake = FakePredictor(probability if probability is not None else 0.0)
        self.available = probability is not None

    def status(self) -> ModelStatus:
        if not self.available:
            return ModelStatus("NOT_FOUND", "No trained model was found.")
        return ModelStatus(AVAILABLE, "A compatible trained model is loaded.", feature_lists.extractor_version(), self.sha, False)

    def predictor(self):
        return self.fake if self.available else None


# ---------------------------------------------------------------- app and client
@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        secret_key=SECRET,
        database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
        upload_tmp_dir=tmp_path / "uploads",
        model_dir=tmp_path / "models",
        apk_library_dir=tmp_path / "apk_library",
        max_apk_size_mb=1,
        min_apk_size_bytes=512,
        analysis_timeout_seconds=60,
        login_max_failures=3,
        scan_queue_wait_seconds=0,
    )


def in_process_extract(path, limits=None, timeout_seconds=0):
    """Same extractor, run in the test process (fast). A few tests use the real isolated worker."""
    return extract_features(path, limits)


@pytest.fixture
def model_service() -> FakeModelService:
    return FakeModelService(0.9)


@pytest.fixture
def make_client(settings) -> Callable[..., TestClient]:
    opened: list[TestClient] = []

    def _make(*, model_service=None, extract_fn=in_process_extract, init=True, raise_exceptions=True, settings_override=None):
        app = create_app(settings_override or settings, model_service=model_service, extract_fn=extract_fn)
        if init:
            init_db(app.state.engine)
        client = TestClient(app, raise_server_exceptions=raise_exceptions)
        client.__enter__()
        opened.append(client)
        return client

    yield _make
    for c in opened:
        c.__exit__(None, None, None)


@pytest.fixture
def client(make_client, model_service) -> TestClient:
    return make_client(model_service=model_service)


# ---------------------------------------------------------------- helpers
def register(client: TestClient, email: str = "alice@example.com", password: str = PASSWORD, name: str = "Alice"):
    return client.post("/register", json={"name": name, "email": email, "password": password})


def login(client: TestClient, email: str = "alice@example.com", password: str = PASSWORD):
    return client.post("/login", json={"email": email, "password": password})


def auth_headers(client: TestClient, email: str = "alice@example.com", password: str = PASSWORD) -> dict[str, str]:
    assert register(client, email, password).status_code in (201, 409)
    response = login(client, email, password)
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def upload(client: TestClient, headers: dict, data: bytes, filename: str = "sample.apk"):
    return client.post(
        "/scan", headers=headers,
        files={"file": (filename, data, "application/vnd.android.package-archive")},
    )


@pytest.fixture
def alice(client) -> dict[str, str]:
    return auth_headers(client, "alice@example.com")


@pytest.fixture
def bob(client) -> dict[str, str]:
    return auth_headers(client, "bob@example.com")


@pytest.fixture
def apk_bytes(make_apk) -> Callable[..., bytes]:
    def _bytes(**kwargs) -> bytes:
        return make_apk(**kwargs).read_bytes()
    return _bytes


@pytest.fixture
def valid_apk_bytes(apk_bytes) -> bytes:
    return apk_bytes(
        permissions=["android.permission.INTERNET", "android.permission.SEND_SMS", "android.permission.READ_SMS"],
        method_refs=[
            ("Landroid/telephony/SmsManager;", "sendTextMessage", "V", ("Ljava/lang/String;",)),
            ("Landroid/app/Activity;", "onCreate", "V", ("Landroid/os/Bundle;",)),
        ],
    )


def uploads_left(settings: Settings) -> list[str]:
    d = settings.upload_tmp_dir
    return sorted(p.name for p in d.glob("scan_*")) if d.exists() else []


def make_synthetic_model_dir(model_dir, *, synthetic: bool = True) -> dict:
    """Train a tiny classifier on SYNTHETIC numbers and save it with the real save_bundle()."""
    from app.ml.modeling import tune_and_fit
    from app.ml.predictor import save_bundle
    from app.ml.splitting import dataset_fingerprint, prepare_split
    from tests.test_ml_pipeline import synthetic_frame

    split = prepare_split(synthetic_frame(40), 0.2, 1)
    model, info = tune_and_fit(split.train, seed=1, cv_folds=3, grid={"rf__n_estimators": [10], "rf__max_depth": [3]})
    save_bundle(model, model_dir, info, {"fingerprint": dataset_fingerprint(split.train), "synthetic": synthetic})
    return json.loads((model_dir / "feature_schema.json").read_text())


@pytest.fixture
def real_model_service_factory(settings):
    def _make(allow_synthetic: bool = False) -> ModelService:
        return ModelService(settings.model_dir, allow_synthetic=allow_synthetic)
    return _make
