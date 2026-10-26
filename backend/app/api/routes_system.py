from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from app.api.deps import get_model_service, get_settings
from app.config import PROJECT_ROOT, Settings
from app.database.session import database_status
from app.services.apk_features import ExtractorEnvironmentError, load_androguard
from app.services.model_service import ModelService

router = APIRouter(tags=["system"])

API_VERSION = "0.5.0"


def _find_installer_apk() -> Path:
    candidates = [
        PROJECT_ROOT / "dist" / "APK_Guardian_release.apk",
        PROJECT_ROOT / "dist" / "APK_Guardian_debug.apk",
        PROJECT_ROOT / "frontend" / "build" / "app" / "outputs" / "flutter-apk" / "app-release.apk",
        PROJECT_ROOT / "frontend" / "build" / "app" / "outputs" / "flutter-apk" / "app-debug.apk",
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="APK installer not found.")


@router.get("/", summary="API status")
def root() -> dict[str, Any]:
    return {
        "name": "APK Guardian API",
        "version": API_VERSION,
        "status": "running",
        "docs": "/docs",
        "download": "/download",
        "notice": "Static analysis prototype. Never executes uploaded APKs.",
    }


@router.get("/download", response_class=HTMLResponse, summary="Download landing page for mobile device")
def download_page(request: Request) -> HTMLResponse:
    try:
        apk_file = _find_installer_apk()
        size_mb = f"{apk_file.stat().st_size / (1024 * 1024):.1f} MB"
        status_msg = f"Available ({size_mb})"
    except HTTPException:
        status_msg = "Build pending"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Download APK Guardian</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            background: #0B1120;
            color: #E2E8F0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
        }}
        .card {{
            background: #1E293B;
            border: 1px solid rgba(56, 189, 248, 0.25);
            border-radius: 20px;
            padding: 32px 24px;
            max-width: 440px;
            width: 90%;
            text-align: center;
            box-shadow: 0 20px 40px rgba(0,0,0,0.5);
        }}
        .shield {{
            width: 72px;
            height: 72px;
            margin: 0 auto 16px;
            background: rgba(6, 182, 212, 0.15);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 2px solid #06B6D4;
        }}
        h1 {{
            font-size: 24px;
            margin: 0 0 8px;
            color: #F8FAFC;
            letter-spacing: 0.5px;
        }}
        .subtitle {{
            font-size: 14px;
            color: #94A3B8;
            margin: 0 0 24px;
            line-height: 1.5;
        }}
        .btn {{
            display: inline-block;
            background: linear-gradient(135deg, #06B6D4 0%, #3B82F6 100%);
            color: #FFFFFF;
            text-decoration: none;
            padding: 16px 28px;
            border-radius: 12px;
            font-weight: 700;
            font-size: 16px;
            box-shadow: 0 4px 14px rgba(6, 182, 212, 0.4);
            transition: transform 0.15s ease;
        }}
        .btn:active {{
            transform: scale(0.97);
        }}
        .info {{
            margin-top: 24px;
            font-size: 12px;
            color: #64748B;
            border-top: 1px solid rgba(255,255,255,0.08);
            padding-top: 16px;
        }}
        .badge {{
            display: inline-block;
            background: rgba(34, 197, 94, 0.15);
            color: #4ADE80;
            border: 1px solid rgba(34, 197, 94, 0.3);
            border-radius: 6px;
            padding: 4px 8px;
            font-size: 12px;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>
    <div class="card">
        <div class="shield">
            <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#06B6D4" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
            </svg>
        </div>
        <h1>APK Guardian</h1>
        <p class="subtitle">Explainable AI-Assisted Android Malware Detection</p>
        <div class="badge">{status_msg}</div>
        <div>
            <a href="/download/apk" class="btn">Download Android APK</a>
        </div>
        <div class="info">
            After downloading, open the APK file on your Android device to install.<br>
            Requires Android 7.0 (API 24) or higher.
        </div>
    </div>
</body>
</html>
"""
    return HTMLResponse(content=html_content)


@router.get("/download/apk", summary="Direct download for APK Guardian Android package")
@router.get("/app.apk", summary="Direct download alias for APK")
def download_apk() -> FileResponse:
    apk_file = _find_installer_apk()
    return FileResponse(
        path=apk_file,
        filename="APK_Guardian.apk",
        media_type="application/vnd.android.package-archive",
    )


@router.get("/health", summary="Health check with dependency and model status")
def health(
    request: Request,
    settings: Settings = Depends(get_settings),
    model_service: ModelService = Depends(get_model_service),
) -> JSONResponse:
    db_state = database_status(request.app.state.engine)
    try:
        engine_info: dict[str, Any] = {"available": True, "androguard_version": load_androguard()[2]}
    except ExtractorEnvironmentError:
        engine_info = {"available": False, "androguard_version": None}
    model = model_service.status().public_dict()

    if db_state != "ok" or not engine_info["available"]:
        overall, code = "unavailable", 503
    elif not model["available"]:
        overall, code = "degraded", 200
    else:
        overall, code = "ok", 200
    body = {
        "status": overall,
        "database": db_state,
        "analysis_engine": engine_info,
        "model": model,
        "limits": {
            "max_apk_size_mb": settings.max_apk_size_mb,
            "analysis_timeout_seconds": settings.analysis_timeout_seconds,
        },
    }
    return JSONResponse(status_code=code, content=body)
