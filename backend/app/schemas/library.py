from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class LibrarySampleItem(BaseModel):
    id: int
    filename: str
    sha256: str
    file_size: int
    relative_path: str
    ground_truth_label: str
    label_source: str
    added_at: str
    last_scan_id: Optional[int] = None
    last_scan_status: str = "NOT_SCANNED"
    last_prediction: Optional[str] = None
    last_risk_score: Optional[int] = None
    last_risk_level: Optional[str] = None
    last_scan_date: Optional[str] = None


class LibraryListResponse(BaseModel):
    samples: list[LibrarySampleItem]
    total: int
    benign_count: int
    malicious_count: int
    unclassified_count: int


class LibraryImportResponse(BaseModel):
    sample: LibrarySampleItem
    is_duplicate: bool
    message: str


class LibraryDeleteResponse(BaseModel):
    sample_id: int
    deleted: bool
    message: str
