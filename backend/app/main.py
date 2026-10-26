"""Application factory.

Run with:  python -m uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000
(a factory is used so the server refuses to start, with a clear message, when SECRET_KEY is missing).
"""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import api_router
from app.api.routes_system import API_VERSION
from app.config import Settings, get_settings
from app.database.session import database_status, make_engine, make_session_factory
from app.errors import install_error_handlers
from app.middleware import RequestGuardMiddleware
from app.security.rate_limit import LoginRateLimiter
from app.services.model_service import ModelService
from app.services.upload import cleanup_stale_uploads

logger = logging.getLogger("apkguardian")


def create_app(
    settings: Optional[Settings] = None,
    *,
    model_service: Optional[ModelService] = None,
    extract_fn: Optional[Callable[..., Any]] = None,
) -> FastAPI:
    """Build the app. `model_service` and `extract_fn` exist so tests can substitute fakes."""
    settings = settings or get_settings()
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    engine = make_engine(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        removed = cleanup_stale_uploads(settings.upload_tmp_dir)
        if removed:
            logger.info("removed %d stale temporary upload(s)", removed)
        from app.services.library_service import ensure_library_directories, sync_library_from_disk
        ensure_library_directories(settings.apk_library_dir)
        state = database_status(engine)
        if state != "ok":
            logger.warning("database is '%s'. Run: python -m app.cli init-db", state)
        else:
            try:
                with app.state.session_factory() as sess:
                    sync_library_from_disk(sess, settings)
            except Exception as exc:
                logger.warning("library sync warning: %s", exc)
        status = app.state.model_service.status()
        logger.info("model status: %s", status.state)
        yield
        engine.dispose()

    app = FastAPI(
        title="APK Guardian API",
        version=API_VERSION,
        description=(
            "Static analysis prototype for Android APKs. Uploaded files are never executed. "
            "Model scores and risk levels are not certainty; see each report's limitations."
        ),
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.engine = engine
    app.state.session_factory = make_session_factory(engine)
    app.state.model_service = model_service or ModelService(settings.model_dir, settings.allow_synthetic_model)
    app.state.login_limiter = LoginRateLimiter(settings.login_max_failures, settings.login_lockout_seconds)
    app.state.scan_semaphore = threading.BoundedSemaphore(settings.max_concurrent_scans)
    app.state.extract_fn = extract_fn

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,  # tokens travel in the Authorization header, not cookies
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )
    app.add_middleware(
        RequestGuardMiddleware,  # added last = outermost: size limits apply before anything else
        upload_paths={"/scan", "/scan/", "/library/import", "/library/import/"},
        upload_limit=settings.max_apk_size_bytes,
        json_limit=settings.json_body_limit_bytes,
    )
    install_error_handlers(app)
    app.include_router(api_router)
    return app
