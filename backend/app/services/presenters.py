"""Turn database rows into API responses. Only curated fields leave the server: internal
exception text, file paths, and raw stored JSON are never returned."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

from app.database.models import Report, Scan
from app.schemas.scans import (
    ExtractionOut,
    FeatureValues,
    FindingOut,
    IndicatorsOut,
    ModelInfoOut,
    PredictionOut,
    ReportOut,
    RiskComponentOut,
    RiskOut,
    ScanListItem,
    ScanOut,
)

SAFE_ERROR_MESSAGES = {
    "FILE_NOT_FOUND": "The uploaded file could not be read.",
    "FILE_TOO_SMALL": "The file is too small to be a valid APK.",
    "FILE_TOO_LARGE": "The file exceeds the size limit.",
    "NOT_A_ZIP": "The file is not a ZIP/APK archive.",
    "CORRUPT_ZIP": "The archive is damaged or incomplete.",
    "ZIP_LIMITS_EXCEEDED": "The archive exceeds the limits for safe analysis.",
    "MISSING_MANIFEST": "The archive contains no AndroidManifest.xml.",
    "MISSING_DEX": "The archive contains no classes.dex, so code-level features are unavailable.",
    "MANIFEST_PARSE_ERROR": "The manifest could not be parsed.",
    "DEX_PARSE_ERROR": "The DEX code could not be parsed or failed its integrity check, so code-level features are unavailable.",
    "TIMEOUT": "Analysis exceeded the time limit.",
    "WORKER_CRASH": "The analysis process ended unexpectedly.",
    "UNEXPECTED_ERROR": "An unexpected error occurred during analysis.",
}


def as_utc(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)  # SQLite drops the zone


MAX_LISTED_NAMES = 500
MAX_NAME_LENGTH = 300


def _names(value: object) -> Optional[list[str]]:
    """Curated copy of a stored name list: strings only, bounded count and length. None = unknown."""
    if not isinstance(value, list):
        return None
    return [str(v)[:MAX_NAME_LENGTH] for v in value[:MAX_LISTED_NAMES] if isinstance(v, str)]


def _indicators(raw: object) -> Optional[IndicatorsOut]:
    if not isinstance(raw, dict) or "permissions" not in raw:
        return None  # extraction failed before the manifest was read, or an older stored scan
    return IndicatorsOut(
        permissions=_names(raw.get("permissions")),
        dangerous_permissions=_names(raw.get("dangerous_permission_names")),
        activities=_names(raw.get("activities")),
        services=_names(raw.get("services")),
        receivers=_names(raw.get("receivers")),
        suspicious_apis=_names(raw.get("suspicious_api_matches")),
    )


def present_scan(scan: Scan) -> ScanOut:
    stored = json.loads(scan.result_json)
    extraction = stored["extraction"]
    assessment = stored["assessment"]
    prediction: Optional[dict] = stored.get("prediction")
    code = extraction.get("error_code")
    report_obj = getattr(scan, "report", None)
    return ScanOut(
        id=scan.id,
        apk_name=scan.apk_name,
        package_name=scan.package_name,
        sha256=scan.sha256,
        file_size=scan.file_size,
        scan_date=as_utc(scan.scan_date),
        scan_status=scan.scan_status,
        extraction=ExtractionOut(
            status=extraction.get("status"),
            error_code=code,
            message=SAFE_ERROR_MESSAGES.get(code, "Analysis was incomplete.") if code else None,
        ),
        features=FeatureValues(**extraction["features"]),
        indicators=_indicators(extraction.get("indicators")),
        classification=scan.classification,
        classification_reasons=assessment["reasons"],
        prediction=PredictionOut(**prediction) if prediction else None,
        model=ModelInfoOut(**stored["model"]),
        risk=RiskOut(
            score=scan.risk_score,
            level=scan.risk_level,
            basis=scan.risk_basis,
            logic_version=scan.risk_logic_version,
            components=[RiskComponentOut(**c) for c in assessment["components"]],
        ),
        reused_previous_result=scan.reused_previous_result,
        explanation=report_obj.explanation if report_obj else None,
        explanation_source=report_obj.explanation_source if report_obj else None,
    )


def present_list_item(scan: Scan) -> ScanListItem:
    return ScanListItem(
        id=scan.id,
        apk_name=scan.apk_name,
        package_name=scan.package_name,
        sha256=scan.sha256,
        scan_date=as_utc(scan.scan_date),
        scan_status=scan.scan_status,
        classification=scan.classification,
        risk_score=scan.risk_score,
        risk_level=scan.risk_level,
    )


def present_report(scan: Scan, report: Report) -> ReportOut:
    return ReportOut(
        scan=present_scan(scan),
        explanation=report.explanation,
        explanation_source=report.explanation_source,
        findings=[FindingOut(**f) for f in json.loads(report.findings)],
        recommendations=json.loads(report.recommendations),
        limitations=json.loads(report.limitations),
    )
