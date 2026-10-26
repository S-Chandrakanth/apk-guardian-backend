from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class FeatureValues(BaseModel):
    """The eight static features. null means 'could not be read', never 'zero'."""

    permissions_count: Optional[int] = None
    activities_count: Optional[int] = None
    services_count: Optional[int] = None
    receivers_count: Optional[int] = None
    api_calls_count: Optional[int] = Field(None, description="Distinct external method references (static), not runtime calls")
    strings_count: Optional[int] = None
    dangerous_permissions: Optional[int] = None
    suspicious_api_calls: Optional[int] = None


class ExtractionOut(BaseModel):
    status: Optional[str] = Field(None, description="OK, PARTIAL, FAILED")
    error_code: Optional[str] = None
    message: Optional[str] = None


class ModelInfoOut(BaseModel):
    status: str = Field(description="AVAILABLE, NOT_FOUND, INCOMPATIBLE, SYNTHETIC_BLOCKED or ERROR")
    available: bool
    detail: str
    extractor_version: Optional[str] = None
    model_id: Optional[str] = Field(None, description="First 12 characters of the model file's SHA-256")
    trained_on_synthetic_test_data: bool = False


class PredictionOut(BaseModel):
    """ML output. Evidence label: ML. Model scores, not certainty."""

    label: str = Field(description="Raw model label at a 0.5 cut-off (BENIGN or MALWARE); see `classification` for the final result")
    benign_probability: float
    malicious_probability: float


class RiskComponentOut(BaseModel):
    name: str
    label: str = Field(description="ML or HEURISTIC")
    points: Optional[float] = Field(None, description="null = evidence unavailable")
    max_points: float
    evidence: list[str]


class RiskOut(BaseModel):
    """Heuristic risk. Evidence label: HEURISTIC (plus ML points when a model score exists). Not a probability of infection."""

    score: Optional[int] = Field(None, ge=0, le=100)
    level: Optional[str] = None
    basis: Optional[str] = Field(None, description="ML_AND_HEURISTIC, HEURISTIC_ONLY or HEURISTIC_PARTIAL")
    logic_version: str
    components: list[RiskComponentOut]


class IndicatorsOut(BaseModel):
    """Names read from the APK. A list is null (not empty) when it could not be read."""

    permissions: Optional[list[str]] = Field(None, description="All requested permission names")
    dangerous_permissions: Optional[list[str]] = Field(None, description="Those on the pinned dangerous-permission list")
    activities: Optional[list[str]] = None
    services: Optional[list[str]] = None
    receivers: Optional[list[str]] = None
    suspicious_apis: Optional[list[str]] = Field(
        None, description="Framework methods referenced by the code that match the pinned watch-list (null if the DEX could not be read)"
    )


class ScanOut(BaseModel):
    id: int
    apk_name: str
    package_name: Optional[str]
    sha256: str
    file_size: int
    scan_date: datetime
    scan_status: str = Field(description="COMPLETED, MODEL_UNAVAILABLE, EXTRACTION_PARTIAL, EXTRACTION_FAILED or PREDICTION_FAILED")
    extraction: ExtractionOut
    features: FeatureValues
    indicators: Optional[IndicatorsOut] = Field(None, description="Names behind the counts; null for scans that could not be read")
    classification: Optional[str] = Field(None, description="BENIGN, SUSPICIOUS, MALICIOUS or NEEDS REVIEW")
    classification_reasons: list[str]
    prediction: Optional[PredictionOut]
    model: ModelInfoOut
    risk: RiskOut
    reused_previous_result: bool
    explanation: Optional[str] = None
    explanation_source: Optional[str] = None
    notice: str = (
        "Static analysis prototype. Model scores and risk levels are not certainty; "
        "the APK was never executed."
    )


class ScanSummaryOut(BaseModel):
    """Counts over ALL of the caller's scans, computed by the database."""

    total: int
    by_status: dict[str, int] = Field(description="Every scan_status key is always present (0 when none)")
    by_classification: dict[str, int] = Field(description="BENIGN, SUSPICIOUS, MALICIOUS, NEEDS REVIEW; always present")
    by_risk_level: dict[str, int] = Field(description="LOW, MEDIUM, HIGH, CRITICAL; scans without a score are not counted")
    last_scan_at: Optional[datetime] = None


class ScanListItem(BaseModel):
    id: int
    apk_name: str
    package_name: Optional[str]
    sha256: str
    scan_date: datetime
    scan_status: str
    classification: Optional[str]
    risk_score: Optional[int]
    risk_level: Optional[str]


class ScanListOut(BaseModel):
    items: list[ScanListItem]
    total: int
    limit: int
    offset: int


class FindingOut(BaseModel):
    label: str = Field(description="OBSERVED, ML or HEURISTIC")
    text: str


class ReportOut(BaseModel):
    scan: ScanOut
    explanation: str
    explanation_source: str = Field(description="rule_based (fixed rules) or ai (a later phase)")
    findings: list[FindingOut]
    recommendations: list[str]
    limitations: list[str]
    evidence_labels: dict[str, str] = {
        "OBSERVED": "Read directly from the APK.",
        "ML": "Output of the trained classifier.",
        "HEURISTIC": "A hand-written rule in this application.",
    }
