from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.config import Settings
from app.database.models import LibrarySample, Scan, User
from app.schemas.library import LibraryListResponse, LibrarySampleItem
from app.services.scan_service import perform_scan
from app.services.upload import SavedUpload, sanitize_filename

logger = logging.getLogger("apkguardian.library")


def ensure_library_directories(library_dir: Path) -> None:
    """Ensure data/apk_library/benign, malicious, and unclassified directories exist."""
    library_dir.mkdir(parents=True, exist_ok=True)
    for category in ("benign", "malicious", "unclassified"):
        (library_dir / category).mkdir(parents=True, exist_ok=True)


def calculate_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sync_library_from_disk(session: Session, settings: Settings) -> None:
    """Scans the library directory on startup or refresh and indexes any real APKs."""
    ensure_library_directories(settings.apk_library_dir)
    found_hashes: set[str] = set()

    for category in ("benign", "malicious", "unclassified"):
        cat_dir = settings.apk_library_dir / category
        if not cat_dir.is_dir():
            continue
        for file_path in cat_dir.glob("*.apk"):
            if not file_path.is_file():
                continue
            try:
                data = file_path.read_bytes()
                file_hash = calculate_sha256(data)
                found_hashes.add(file_hash)
                rel_path = str(file_path.relative_to(settings.apk_library_dir).as_posix())

                existing = session.scalar(select(LibrarySample).where(LibrarySample.sha256 == file_hash))
                if not existing:
                    sample = LibrarySample(
                        filename=file_path.name,
                        sha256=file_hash,
                        file_size=len(data),
                        relative_path=rel_path,
                        ground_truth_label=category,
                        label_source="directory_structure",
                        last_scan_status="NOT_SCANNED",
                    )
                    session.add(sample)
                else:
                    existing.relative_path = rel_path
                    existing.file_size = len(data)
            except Exception as e:
                logger.warning("Failed to index sample %s: %s", file_path, e)

    session.commit()


def get_library_samples(
    session: Session,
    settings: Settings,
    search: Optional[str] = None,
    label: Optional[str] = None,
    scan_status: Optional[str] = None,
    sort_by: str = "date",
) -> LibraryListResponse:
    ensure_library_directories(settings.apk_library_dir)
    query = select(LibrarySample)

    if search:
        search_term = f"%{search.strip().lower()}%"
        query = query.where(
            func.lower(LibrarySample.filename).like(search_term)
            | func.lower(LibrarySample.sha256).like(search_term)
        )

    if label and label.lower() != "all":
        query = query.where(func.lower(LibrarySample.ground_truth_label) == label.lower())

    if scan_status and scan_status.lower() != "all":
        query = query.where(func.lower(LibrarySample.last_scan_status) == scan_status.lower())

    if sort_by == "filename":
        query = query.order_by(LibrarySample.filename.asc())
    elif sort_by == "size":
        query = query.order_by(LibrarySample.file_size.desc())
    else:  # date
        query = query.order_by(LibrarySample.added_at.desc())

    items = list(session.scalars(query).all())

    # Calculate counts
    all_samples = list(session.scalars(select(LibrarySample)).all())
    benign_count = sum(1 for s in all_samples if s.ground_truth_label.lower() == "benign")
    malicious_count = sum(1 for s in all_samples if s.ground_truth_label.lower() == "malicious")
    unclassified_count = sum(1 for s in all_samples if s.ground_truth_label.lower() == "unclassified")

    sample_items = [
        LibrarySampleItem(
            id=s.id,
            filename=s.filename,
            sha256=s.sha256,
            file_size=s.file_size,
            relative_path=s.relative_path,
            ground_truth_label=s.ground_truth_label,
            label_source=s.label_source,
            added_at=s.added_at.isoformat(),
            last_scan_id=s.last_scan_id,
            last_scan_status=s.last_scan_status,
            last_prediction=s.last_prediction,
            last_risk_score=s.last_risk_score,
            last_risk_level=s.last_risk_level,
            last_scan_date=s.last_scan_date.isoformat() if s.last_scan_date else None,
        )
        for s in items
    ]

    return LibraryListResponse(
        samples=sample_items,
        total=len(sample_items),
        benign_count=benign_count,
        malicious_count=malicious_count,
        unclassified_count=unclassified_count,
    )


def import_sample_to_library(
    session: Session,
    settings: Settings,
    filename: str,
    file_bytes: bytes,
    label: Optional[str] = None,
    label_source: Optional[str] = None,
) -> tuple[LibrarySampleItem, bool, str]:
    ensure_library_directories(settings.apk_library_dir)
    safe_name = sanitize_filename(filename)
    if not safe_name.lower().endswith(".apk"):
        safe_name = f"{safe_name}.apk"

    file_hash = calculate_sha256(file_bytes)

    # Check for duplicate
    existing = session.scalar(select(LibrarySample).where(LibrarySample.sha256 == file_hash))
    if existing:
        item = LibrarySampleItem(
            id=existing.id,
            filename=existing.filename,
            sha256=existing.sha256,
            file_size=existing.file_size,
            relative_path=existing.relative_path,
            ground_truth_label=existing.ground_truth_label,
            label_source=existing.label_source,
            added_at=existing.added_at.isoformat(),
            last_scan_id=existing.last_scan_id,
            last_scan_status=existing.last_scan_status,
            last_prediction=existing.last_prediction,
            last_risk_score=existing.last_risk_score,
            last_risk_level=existing.last_risk_level,
            last_scan_date=existing.last_scan_date.isoformat() if existing.last_scan_date else None,
        )
        return item, True, f"Sample with hash {file_hash[:8]} already exists in library."

    category = "unclassified"
    if label and label.lower() in ("benign", "malicious"):
        category = label.lower()

    dest_dir = settings.apk_library_dir / category
    dest_path = dest_dir / safe_name
    
    # Avoid filename collisions
    if dest_path.exists():
        stem = Path(safe_name).stem
        dest_path = dest_dir / f"{stem}_{file_hash[:8]}.apk"

    dest_path.write_bytes(file_bytes)
    rel_path = str(dest_path.relative_to(settings.apk_library_dir).as_posix())

    sample = LibrarySample(
        filename=dest_path.name,
        sha256=file_hash,
        file_size=len(file_bytes),
        relative_path=rel_path,
        ground_truth_label=category,
        label_source=label_source or ("user_provided" if label else "unclassified"),
        last_scan_status="NOT_SCANNED",
    )
    session.add(sample)
    session.commit()
    session.refresh(sample)

    item = LibrarySampleItem(
        id=sample.id,
        filename=sample.filename,
        sha256=sample.sha256,
        file_size=sample.file_size,
        relative_path=sample.relative_path,
        ground_truth_label=sample.ground_truth_label,
        label_source=sample.label_source,
        added_at=sample.added_at.isoformat(),
        last_scan_id=sample.last_scan_id,
        last_scan_status=sample.last_scan_status,
        last_prediction=sample.last_prediction,
        last_risk_score=sample.last_risk_score,
        last_risk_level=sample.last_risk_level,
        last_scan_date=None,
    )
    return item, False, "Sample imported to library successfully."


def delete_library_sample(session: Session, settings: Settings, sample_id: int) -> tuple[bool, str]:
    sample = session.get(LibrarySample, sample_id)
    if not sample:
        return False, "Sample not found in library."

    # Validate path safely inside apk_library_dir to prevent path traversal
    file_path = (settings.apk_library_dir / sample.relative_path).resolve()
    library_root = settings.apk_library_dir.resolve()
    try:
        file_path.relative_to(library_root)
        if file_path.is_file():
            file_path.unlink()
    except (ValueError, OSError) as e:
        logger.warning("Error deleting sample file %s: %s", file_path, e)

    session.delete(sample)
    session.commit()
    return True, f"Sample '{sample.filename}' removed from library."


def scan_library_sample(
    session: Session,
    settings: Settings,
    sample_id: int,
    user: User,
    model_service: Any,
    scan_semaphore: Any,
    extract_fn: Optional[Callable[..., Any]] = None,
) -> Any:
    sample = session.get(LibrarySample, sample_id)
    if not sample:
        raise ValueError("Library sample not found")

    file_path = (settings.apk_library_dir / sample.relative_path).resolve()
    library_root = settings.apk_library_dir.resolve()
    file_path.relative_to(library_root)  # Safety check

    if not file_path.is_file():
        sample.last_scan_status = "FILE_MISSING"
        session.commit()
        raise ValueError("Sample APK file is missing from disk")

    data = file_path.read_bytes()
    file_hash = calculate_sha256(data)

    if not scan_semaphore.acquire(timeout=settings.scan_queue_wait_seconds):
        raise RuntimeError("The server is busy analysing other files. Try again shortly.")

    try:
        saved = SavedUpload(path=file_path, size=len(data), sha256=file_hash)
        scan_record = perform_scan(
            db=session,
            user=user,
            saved=saved,
            display_name=sample.filename,
            settings=settings,
            model_service=model_service,
            extract_fn=extract_fn,
        )
    finally:
        scan_semaphore.release()

    # Sync back into LibrarySample
    sample.last_scan_id = scan_record.id
    sample.last_scan_status = scan_record.scan_status
    sample.last_prediction = scan_record.prediction
    sample.last_risk_score = scan_record.risk_score
    sample.last_risk_level = scan_record.risk_level
    sample.last_scan_date = scan_record.scan_date
    session.commit()

    return scan_record
