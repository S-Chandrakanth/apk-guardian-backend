"""Error responses that never expose passwords, tokens, secrets, or internal exception text."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger("apkguardian.errors")


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI's default 422 body echoes the submitted value ("input"), which would
        # include a password that failed a length rule. Return only where and why.
        problems = [
            {"loc": [str(part) for part in err.get("loc", ())], "msg": str(err.get("msg", "Invalid value")), "type": str(err.get("type", ""))}
            for err in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": problems})

    @app.exception_handler(SQLAlchemyError)
    async def database_error(request: Request, exc: SQLAlchemyError) -> JSONResponse:
        logger.error("database error (request %s)", _request_id(request), exc_info=exc)
        return JSONResponse(
            status_code=503,
            content={"detail": "The database is unavailable or returned an error. Try again later.", "request_id": _request_id(request)},
        )

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled error (request %s)", _request_id(request), exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error.", "request_id": _request_id(request)},
        )
