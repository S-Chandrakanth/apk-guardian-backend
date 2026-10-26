"""Loads the trained classifier once and tells the API whether it can be used.

This wraps the existing app.ml.predictor.Predictor (which already validates the schema,
feature order, extractor version and model checksum). It adds no ML logic of its own.
A model that is missing, incompatible, or trained on synthetic test data is reported as
unavailable. Nothing here ever trains a model or fabricates a prediction.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.ml.predictor import MODEL_FILENAME, SCHEMA_FILENAME, ModelSchemaError, Predictor

logger = logging.getLogger("apkguardian.model")

NOT_FOUND = "NOT_FOUND"
INCOMPATIBLE = "INCOMPATIBLE"
SYNTHETIC_BLOCKED = "SYNTHETIC_BLOCKED"
ERROR = "ERROR"
AVAILABLE = "AVAILABLE"

_DETAIL = {
    AVAILABLE: "A compatible trained model is loaded.",
    NOT_FOUND: "No trained model was found. Train one with ml_training\\train_model.py on a validated, real dataset.",
    INCOMPATIBLE: (
        "A model exists but is incompatible with the current feature schema or extractor version, "
        "or failed its integrity check. Re-extract the dataset and retrain."
    ),
    SYNTHETIC_BLOCKED: (
        "The saved model was trained on synthetic test data and is blocked. "
        "Train on a real, labelled dataset."
    ),
    ERROR: "The model could not be loaded.",
}


@dataclass(frozen=True)
class ModelStatus:
    state: str
    detail: str
    extractor_version: Optional[str] = None
    model_sha256: Optional[str] = None
    trained_on_synthetic_test_data: bool = False

    @property
    def available(self) -> bool:
        return self.state == AVAILABLE

    def public_dict(self) -> dict[str, object]:
        return {
            "status": self.state,
            "available": self.available,
            "detail": self.detail,
            "extractor_version": self.extractor_version,
            "model_id": self.model_sha256[:12] if self.model_sha256 else None,
            "trained_on_synthetic_test_data": self.trained_on_synthetic_test_data,
        }


class ModelService:
    def __init__(self, model_dir: Path, allow_synthetic: bool = False, retry_seconds: float = 5.0) -> None:
        self.model_dir = Path(model_dir)
        self.allow_synthetic = allow_synthetic
        self.retry_seconds = retry_seconds
        self._lock = threading.Lock()
        self._predictor: Optional[Predictor] = None
        self._status = ModelStatus(NOT_FOUND, _DETAIL[NOT_FOUND])
        self._signature: Optional[tuple[int, int]] = None
        self._last_attempt = 0.0

    # -- files ---------------------------------------------------------
    def _file_signature(self) -> Optional[tuple[int, int]]:
        try:
            return (
                (self.model_dir / SCHEMA_FILENAME).stat().st_mtime_ns,
                (self.model_dir / MODEL_FILENAME).stat().st_mtime_ns,
            )
        except OSError:
            return None

    def _load(self) -> None:
        self._last_attempt = time.monotonic()
        self._signature = self._file_signature()
        self._predictor = None
        if not (self.model_dir / SCHEMA_FILENAME).is_file():
            self._status = ModelStatus(NOT_FOUND, _DETAIL[NOT_FOUND])
            return
        try:
            predictor = Predictor.load(self.model_dir)
        except ModelSchemaError as exc:
            logger.warning("model rejected: %s", exc)
            self._status = ModelStatus(INCOMPATIBLE, _DETAIL[INCOMPATIBLE])
            return
        except Exception:  # unpickling errors, missing libraries, damaged files ...
            logger.exception("model failed to load")
            self._status = ModelStatus(ERROR, _DETAIL[ERROR])
            return

        synthetic = predictor.trained_on_synthetic_test_data
        if synthetic and not self.allow_synthetic:
            self._status = ModelStatus(
                SYNTHETIC_BLOCKED, _DETAIL[SYNTHETIC_BLOCKED],
                predictor.schema.get("extractor_version"), predictor.schema.get("model_sha256"), True,
            )
            return
        self._predictor = predictor
        self._status = ModelStatus(
            AVAILABLE, _DETAIL[AVAILABLE],
            predictor.schema.get("extractor_version"), predictor.schema.get("model_sha256"), synthetic,
        )

    def _ensure(self) -> None:
        with self._lock:
            current = self._file_signature()
            if self._predictor is not None and current == self._signature:
                return
            changed = current != self._signature
            if not changed and self._last_attempt:
                # Same files as last time give the same outcome, so there is nothing to retry,
                # except after a load ERROR, which may have been transient.
                recently_tried = (time.monotonic() - self._last_attempt) < self.retry_seconds
                if self._status.state != ERROR or recently_tried:
                    return
            self._load()

    # -- public --------------------------------------------------------
    def status(self) -> ModelStatus:
        self._ensure()
        return self._status

    def predictor(self) -> Optional[Predictor]:
        self._ensure()
        return self._predictor
