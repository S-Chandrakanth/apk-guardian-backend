"""Scan orchestration: ties together the EXISTING extractor and predictor with risk, report and storage.

No extraction or model logic is duplicated here. Flow:

  saved upload (already validated and hashed)
    -> reuse a stored result? (same user, same hash, COMPLETED, same model + extractor + schema)
    -> else extract features (isolated worker process, hard timeout)
    -> predict (only if extraction is complete and a compatible model exists)
    -> risk + classification -> rule-based report -> one database transaction
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Optional

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.config import Settings
from app.database.models import Report, Scan, User
from app.ml import feature_lists
from app.ml.schema import FEATURE_NAMES
from app.services import apk_features
from app.services.apk_features import ExtractionLimits, ExtractorEnvironmentError
from app.services.model_service import ModelService
from app.services.report_builder import build_report
from app.services.risk import RISK_LOGIC_VERSION, assess
from app.services.upload import SavedUpload

logger = logging.getLogger("apkguardian.scan")

FEATURE_SCHEMA_HASH = hashlib.sha256(",".join(FEATURE_NAMES).encode("utf-8")).hexdigest()

ExtractFn = Callable[..., Any]


def find_reusable_result(
    db: Session, user_id: int, sha256: str, model_sha256: str, extractor_version: str
) -> Optional[Scan]:
    """A previous COMPLETED scan of the same file by the SAME user, made with the same model,
    extractor version and feature schema. Restricted to the same user so one user can never
    learn that someone else scanned a particular file."""
    return db.execute(
        select(Scan)
        .where(
            Scan.user_id == user_id,
            Scan.sha256 == sha256,
            Scan.scan_status == "COMPLETED",
            Scan.model_sha256 == model_sha256,
            Scan.extractor_version == extractor_version,
            Scan.feature_schema_hash == FEATURE_SCHEMA_HASH,
        )
        .order_by(Scan.id.desc())
        .limit(1)
    ).scalar_one_or_none()


def _scan_status(extraction_status: str, predictor_present: bool, prediction: Optional[dict]) -> str:
    if extraction_status == "FAILED":
        return "EXTRACTION_FAILED"
    if extraction_status == "PARTIAL":
        return "EXTRACTION_PARTIAL"
    if not predictor_present:
        return "MODEL_UNAVAILABLE"
    return "COMPLETED" if prediction is not None else "PREDICTION_FAILED"


def perform_scan(
    db: Session,
    user: User,
    saved: SavedUpload,
    display_name: str,
    settings: Settings,
    model_service: ModelService,
    extract_fn: Optional[ExtractFn] = None,
) -> Scan:
    """Analyse one saved upload and store the result. Raises ExtractorEnvironmentError if the
    analysis engine is unusable, and SQLAlchemyError if the result cannot be stored."""
    extract_fn = extract_fn or apk_features.extract_features_isolated
    model_status = model_service.status()
    predictor = model_service.predictor() if model_status.available else None
    extractor_version = feature_lists.extractor_version()

    reused = False
    extraction: dict[str, Any]
    prediction: Optional[dict[str, Any]] = None

    cached = (
        find_reusable_result(db, user.id, saved.sha256, model_status.model_sha256 or "", extractor_version)
        if predictor is not None
        else None
    )
    if cached is not None:
        stored = json.loads(cached.result_json)
        extraction, prediction, reused = stored["extraction"], stored["prediction"], True
    else:
        limits = ExtractionLimits(
            min_size_bytes=min(settings.min_apk_size_bytes, 512),
            max_size_bytes=settings.max_apk_size_bytes,
        )
        result = extract_fn(saved.path, limits=limits, timeout_seconds=settings.analysis_timeout_seconds)
        extraction = {
            "status": result.status,
            "features": dict(result.features),
            "package_name": result.package_name,
            "error_code": result.error_code,
            "indicators": result.indicators,
            "extractor_version": result.extractor_version,
        }  # the extractor's free-text error message is deliberately NOT stored or returned

        if extraction["status"] == "OK" and predictor is not None:
            try:
                p = predictor.predict(extraction["features"])
                prediction = {
                    "label": p.raw_label,
                    "benign_probability": p.benign_probability,
                    "malicious_probability": p.malware_probability,
                }
            except Exception:
                logger.exception("prediction failed")
                prediction = None

    status = _scan_status(extraction["status"], predictor is not None, prediction)
    problem = {
        "MODEL_UNAVAILABLE": model_status.detail,
        "PREDICTION_FAILED": "The model could not score this APK.",
    }.get(status)
    assessment = assess(
        extraction["status"],
        extraction["features"] if extraction["status"] != "FAILED" else {},
        extraction.get("indicators", {}),
        prediction["malicious_probability"] if prediction else None,
        problem,
    )
    model_info = model_status.public_dict()
    report = build_report(
        extraction.get("package_name"), extraction["features"], extraction.get("indicators", {}),
        assessment, prediction, model_info, extraction["status"], extraction.get("error_code"),
        settings=settings,
    )

    scan = Scan(
        user_id=user.id,
        apk_name=display_name,
        package_name=extraction.get("package_name"),
        sha256=saved.sha256,
        file_size=saved.size,
        scan_status=status,
        extraction_status=extraction["status"],
        error_code=extraction.get("error_code"),
        classification=assessment.classification,
        prediction=prediction["label"] if prediction else None,
        benign_probability=prediction["benign_probability"] if prediction else None,
        malicious_probability=prediction["malicious_probability"] if prediction else None,
        risk_score=assessment.score,
        risk_level=assessment.level,
        risk_basis=assessment.basis,
        extractor_version=extraction["extractor_version"],
        feature_schema_hash=FEATURE_SCHEMA_HASH,
        model_sha256=model_status.model_sha256 if predictor is not None else None,
        risk_logic_version=RISK_LOGIC_VERSION,
        reused_previous_result=reused,
        result_json=json.dumps(
            {
                "extraction": extraction,
                "prediction": prediction,
                "model": model_info,
                "assessment": asdict(assessment),
            }
        ),
    )
    scan.report = Report(
        explanation=report["explanation"],
        explanation_source=report["explanation_source"],
        findings=json.dumps(report["findings"]),
        recommendations=json.dumps(report["recommendations"]),
        limitations=json.dumps(report["limitations"]),
    )
    try:
        db.add(scan)
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        raise
    db.refresh(scan)
    return scan


__all__ = ["perform_scan", "find_reusable_result", "ExtractorEnvironmentError", "FEATURE_SCHEMA_HASH"]
