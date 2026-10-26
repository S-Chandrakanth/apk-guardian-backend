"""Save, load, and use the trained classifier.

This module only classifies. It knows nothing about explanations, risk levels, or the API.
The AI explanation component is separate and receives this module's output as read-only
input; it can never change a prediction.

It returns raw model scores. It does NOT decide BENIGN / SUSPICIOUS / MALICIOUS /
NEEDS REVIEW. That decision combines the score with other evidence in a later phase.
A probability here is a model score on this dataset's class balance, not a guarantee of
real-world certainty.

Security note: joblib files are pickles. Only load model files you trained yourself. The
SHA-256 check below detects a model file that does not match its schema file; it does not
make an untrusted file safe to load.
"""

from __future__ import annotations

import json
import logging
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import joblib
import pandas as pd
import sklearn

from app.ml import feature_lists
from app.ml.dataset_validation import parse_count
from app.ml.modeling import malware_probability
from app.ml.schema import FEATURE_DEFINITIONS, FEATURE_NAMES, LABEL_BENIGN, LABEL_MALWARE
from app.services.hashing import sha256_file

logger = logging.getLogger(__name__)

MODEL_FILENAME = "rf_baseline.joblib"
SCHEMA_FILENAME = "feature_schema.json"
SCHEMA_VERSION = 1


class ModelSchemaError(RuntimeError):
    """The saved model and its schema file are missing, inconsistent, or incompatible."""


class IncompleteFeaturesError(ValueError):
    """Some features are missing or None. The classifier will not guess them."""


class InvalidFeaturesError(ValueError):
    """A feature value is not a non-negative integer."""


@dataclass(frozen=True)
class Prediction:
    malware_probability: float
    benign_probability: float
    raw_label: str  # simple 0.5 cut-off; NOT the final classification shown to users
    model_file: str
    extractor_version: str
    trained_on_synthetic_test_data: bool


def save_bundle(
    pipeline: Any,
    model_dir: str | Path,
    training_info: Mapping[str, Any],
    dataset_info: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Write the model and its feature schema. Returns (model_path, schema_path)."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path, schema_path = model_dir / MODEL_FILENAME, model_dir / SCHEMA_FILENAME
    joblib.dump(pipeline, model_path)
    schema = {
        "schema_version": SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "feature_definitions": dict(FEATURE_DEFINITIONS),
        "extractor_version": feature_lists.extractor_version(),
        "label_encoding": {LABEL_BENIGN: 0, LABEL_MALWARE: 1},
        "positive_class": LABEL_MALWARE,
        "model_file": MODEL_FILENAME,
        "model_sha256": sha256_file(model_path),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_dataset": dict(dataset_info),
        "training": dict(training_info),
    }
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    return model_path, schema_path


class Predictor:
    def __init__(self, model: Any, schema: dict[str, Any], model_path: Path) -> None:
        self.model = model
        self.schema = schema
        self.model_path = model_path

    @classmethod
    def load(
        cls,
        model_dir: str | Path,
        expected_extractor_version: Optional[str] = None,
    ) -> "Predictor":
        model_dir = Path(model_dir)
        schema_path = model_dir / SCHEMA_FILENAME
        if not schema_path.is_file():
            raise ModelSchemaError(f"No {SCHEMA_FILENAME} in {model_dir}. Train a model first.")
        try:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ModelSchemaError(f"{SCHEMA_FILENAME} is not valid JSON: {exc}") from exc

        if schema.get("schema_version") != SCHEMA_VERSION:
            raise ModelSchemaError("Unsupported schema_version in feature_schema.json.")
        if schema.get("feature_names") != list(FEATURE_NAMES):
            raise ModelSchemaError(
                "The model was trained with a different feature list or order than this code uses."
            )
        expected = expected_extractor_version or feature_lists.extractor_version()
        if schema.get("extractor_version") != expected:
            raise ModelSchemaError(
                f"Model was trained with extractor '{schema.get('extractor_version')}' but the current "
                f"extractor is '{expected}'. Re-extract the dataset and retrain."
            )

        model_path = model_dir / str(schema.get("model_file", MODEL_FILENAME))
        if not model_path.is_file():
            raise ModelSchemaError(f"Model file not found: {model_path}")
        if sha256_file(model_path) != schema.get("model_sha256"):
            raise ModelSchemaError("Model file does not match the SHA-256 recorded in its schema.")

        trained_with = str(schema.get("sklearn_version", ""))
        if trained_with.split(".")[:2] != sklearn.__version__.split(".")[:2]:
            logger.warning(
                "Model was trained with scikit-learn %s but %s is installed; results may differ.",
                trained_with, sklearn.__version__,
            )
        model = joblib.load(model_path)
        predictor = cls(model, schema, model_path)
        if predictor.trained_on_synthetic_test_data:
            logger.warning("This model was trained on SYNTHETIC TEST DATA. Its output means nothing.")
        return predictor

    @property
    def trained_on_synthetic_test_data(self) -> bool:
        return bool(self.schema.get("training_dataset", {}).get("synthetic", False))

    def _vector(self, features: Mapping[str, Any]) -> pd.DataFrame:
        missing = [n for n in FEATURE_NAMES if n not in features]
        if missing:
            raise IncompleteFeaturesError(f"Missing features: {', '.join(missing)}")
        blank = [n for n in FEATURE_NAMES if features[n] is None]
        if blank:
            raise IncompleteFeaturesError(
                f"Features not available (extraction incomplete): {', '.join(blank)}"
            )
        values: list[int] = []
        for name in FEATURE_NAMES:
            parsed = parse_count(features[name])
            if parsed is None:
                raise InvalidFeaturesError(f"Feature {name} must be a non-negative integer, got {features[name]!r}")
            values.append(parsed)
        return pd.DataFrame([values], columns=list(FEATURE_NAMES), dtype="int64")

    def predict(self, features: Mapping[str, Any]) -> Prediction:
        proba = float(malware_probability(self.model, self._vector(features))[0])
        return Prediction(
            malware_probability=proba,
            benign_probability=1.0 - proba,
            raw_label=LABEL_MALWARE if proba >= 0.5 else LABEL_BENIGN,
            model_file=self.model_path.name,
            extractor_version=str(self.schema["extractor_version"]),
            trained_on_synthetic_test_data=self.trained_on_synthetic_test_data,
        )
