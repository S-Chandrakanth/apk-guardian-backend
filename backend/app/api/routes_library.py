from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db, get_model_service, get_settings
from app.config import Settings
from app.database.models import User
from app.schemas.library import LibraryDeleteResponse, LibraryImportResponse, LibraryListResponse
from app.schemas.scans import ScanOut
from app.services.library_service import (
    delete_library_sample,
    get_library_samples,
    import_sample_to_library,
    scan_library_sample,
)
from app.services.presenters import present_scan

router = APIRouter(prefix="/library", tags=["library"])


@router.get("", response_model=LibraryListResponse)
def list_library_samples(
    search: Optional[str] = Query(None, description="Search by filename or hash"),
    label: Optional[str] = Query(None, description="Filter by label (benign, malicious, unclassified)"),
    scan_status: Optional[str] = Query(None, description="Filter by scan status"),
    sort_by: str = Query("date", description="Sort by 'date', 'filename', or 'size'"),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    user: User = Depends(get_current_user),
) -> LibraryListResponse:
    """Lists APK samples stored in the configured library."""
    return get_library_samples(
        session=db,
        settings=settings,
        search=search,
        label=label,
        scan_status=scan_status,
        sort_by=sort_by,
    )


@router.post("/import", response_model=LibraryImportResponse)
async def import_apk(
    file: UploadFile = File(...),
    label: Optional[str] = Form(None),
    label_source: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    user: User = Depends(get_current_user),
) -> LibraryImportResponse:
    """Imports a real APK into the library. Deduplicates by SHA-256 hash."""
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="A filename is required.")
    
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")
    if len(content) > settings.max_apk_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {settings.max_apk_size_mb} MB.",
        )

    sample, is_dup, msg = import_sample_to_library(
        session=db,
        settings=settings,
        filename=file.filename,
        file_bytes=content,
        label=label,
        label_source=label_source,
    )
    return LibraryImportResponse(sample=sample, is_duplicate=is_dup, message=msg)


@router.post("/{sample_id}/scan", response_model=ScanOut)
def scan_sample(
    sample_id: int,
    request: Request,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    user: User = Depends(get_current_user),
    model_service=Depends(get_model_service),
) -> ScanOut:
    """Triggers static analysis and model inference on a library sample."""
    try:
        scan_record = scan_library_sample(
            session=db,
            settings=settings,
            sample_id=sample_id,
            user=user,
            model_service=model_service,
            scan_semaphore=request.app.state.scan_semaphore,
            extract_fn=getattr(request.app.state, "extract_fn", None),
        )
        return present_scan(scan_record)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Analysis failed: {exc}")


@router.delete("/{sample_id}", response_model=LibraryDeleteResponse)
def delete_sample(
    sample_id: int,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    user: User = Depends(get_current_user),
) -> LibraryDeleteResponse:
    """Safely removes an APK sample from the library without deleting previous scan reports."""
    ok, msg = delete_library_sample(session=db, settings=settings, sample_id=sample_id)
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=msg)
    return LibraryDeleteResponse(sample_id=sample_id, deleted=True, message=msg)


@router.get("/{sample_id}/download", summary="Download sample APK from library")
def download_sample_apk(
    sample_id: int,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    user: User = Depends(get_current_user),
) -> FileResponse:
    from app.database.models import LibrarySample
    sample = db.get(LibrarySample, sample_id)
    if not sample:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sample not found.")

    file_path = (settings.apk_library_dir / sample.relative_path).resolve()
    library_root = settings.apk_library_dir.resolve()
    try:
        file_path.relative_to(library_root)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid path.")

    if not file_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sample file missing from disk.")

    return FileResponse(
        path=file_path,
        filename=sample.filename,
        media_type="application/vnd.android.package-archive",
    )

