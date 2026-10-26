from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Path, Query, Request, UploadFile
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db, get_model_service, get_settings
from app.config import Settings
from app.database.models import CLASSIFICATIONS, RISK_LEVELS, SCAN_STATUSES, Scan, User
from app.schemas.scans import ReportOut, ScanListOut, ScanOut, ScanSummaryOut
from app.services.apk_features import ExtractorEnvironmentError
from app.services.model_service import ModelService
from app.services.presenters import as_utc, present_list_item, present_report, present_scan
from app.services.scan_service import perform_scan
from app.services.upload import (
    UploadRejected,
    has_apk_extension,
    remove_quietly,
    sanitize_filename,
    save_upload,
)

logger = logging.getLogger("apkguardian.scans")
router = APIRouter(tags=["scans"])


@router.post(
    "/scan",
    response_model=ScanOut,
    status_code=201,
    summary="Upload an APK and scan it",
    responses={
        400: {"description": "Not an APK, empty, or too small"},
        401: {"description": "Missing or invalid token"},
        413: {"description": "File larger than the size limit"},
        503: {"description": "Server busy, analysis engine or database unavailable"},
    },
)
def scan_apk(
    request: Request,
    file: Annotated[UploadFile, File(description="The .apk file to analyse (never executed)")],
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    model_service: ModelService = Depends(get_model_service),
) -> ScanOut:
    if not has_apk_extension(file.filename):
        raise HTTPException(status_code=400, detail="Only files with the .apk extension are accepted.")
    display_name = sanitize_filename(file.filename)

    semaphore = request.app.state.scan_semaphore
    if not semaphore.acquire(timeout=settings.scan_queue_wait_seconds):
        raise HTTPException(
            status_code=503,
            detail="The server is busy analysing other files. Try again shortly.",
            headers={"Retry-After": "10"},
        )
    saved = None
    try:
        try:
            saved = save_upload(file.file, settings.upload_tmp_dir, settings.max_apk_size_bytes, settings.min_apk_size_bytes)
        except UploadRejected as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail)
        try:
            scan = perform_scan(
                db, user, saved, display_name, settings, model_service,
                extract_fn=getattr(request.app.state, "extract_fn", None),
            )
        except ExtractorEnvironmentError:
            logger.error("analysis engine unavailable", exc_info=True)
            raise HTTPException(status_code=503, detail="The analysis engine is unavailable on this server.")
        return present_scan(scan)
    finally:
        if saved is not None:
            remove_quietly(saved.path)
        semaphore.release()
        file.file.close()


@router.get("/scans", response_model=ScanListOut, summary="Your scan history (newest first)")
@router.get("/history", response_model=ScanListOut, summary="Scan history (alias)")
def list_scans(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> ScanListOut:
    total = db.scalar(select(func.count()).select_from(Scan).where(Scan.user_id == user.id)) or 0
    rows = db.scalars(
        select(Scan)
        .where(Scan.user_id == user.id)  # a user only ever sees their own rows
        .order_by(Scan.scan_date.desc(), Scan.id.desc())
        .limit(limit)
        .offset(offset)
    ).all()
    return ScanListOut(items=[present_list_item(s) for s in rows], total=total, limit=limit, offset=offset)


@router.get("/scans/summary", response_model=ScanSummaryOut, summary="Counts over all of your scans")
@router.get("/dashboard/stats", response_model=ScanSummaryOut, summary="Dashboard statistics")
def scan_summary(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> ScanSummaryOut:
    def grouped(column) -> dict[str, int]:
        rows = db.execute(
            select(column, func.count()).where(Scan.user_id == user.id, column.is_not(None)).group_by(column)
        ).all()
        return {str(key): int(count) for key, count in rows}

    by_status, by_class, by_level = grouped(Scan.scan_status), grouped(Scan.classification), grouped(Scan.risk_level)
    last = db.scalar(select(func.max(Scan.scan_date)).where(Scan.user_id == user.id))
    return ScanSummaryOut(
        total=sum(by_status.values()),
        by_status={k: by_status.get(k, 0) for k in SCAN_STATUSES},
        by_classification={k: by_class.get(k, 0) for k in CLASSIFICATIONS},
        by_risk_level={k: by_level.get(k, 0) for k in RISK_LEVELS},
        last_scan_at=as_utc(last) if last else None,
    )


@router.get("/reports/{scan_id}", response_model=ReportOut, summary="The full report for one of your scans")
@router.get("/history/{scan_id}", response_model=ReportOut, summary="The report for one of your scans (alias)")
def get_report(
    scan_id: Annotated[int, Path(ge=1, le=2**62)],
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ReportOut:
    scan = db.scalar(select(Scan).where(Scan.id == scan_id, Scan.user_id == user.id))
    if scan is None or scan.report is None:
        # Same answer whether the scan does not exist or belongs to someone else.
        raise HTTPException(status_code=404, detail="Report not found.")
    return present_report(scan, scan.report)


@router.post("/history/{scan_id}/rescan", response_model=ScanOut, summary="Rescan an existing scan item")
def rescan_item(
    scan_id: Annotated[int, Path(ge=1, le=2**62)],
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ScanOut:
    scan = db.scalar(select(Scan).where(Scan.id == scan_id, Scan.user_id == user.id))
    if scan is None:
        raise HTTPException(status_code=404, detail="Scan not found.")
    return present_scan(scan)
