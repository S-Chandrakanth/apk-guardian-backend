"""FastAPI dependencies: settings, database session, model service, current user."""

from __future__ import annotations

from typing import Iterator, Optional

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.config import Settings
from app.database.models import User
from app.security.tokens import decode_access_token
from app.services.model_service import ModelService

bearer_scheme = HTTPBearer(auto_error=False, description="Paste the access_token returned by POST /login")


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_model_service(request: Request) -> ModelService:
    return request.app.state.model_service


def get_db(request: Request) -> Iterator[Session]:
    session = request.app.state.session_factory()
    try:
        yield session
    finally:
        session.close()


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=401,
        detail="Not authenticated.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> User:
    """Identify the caller from the signed token. The user id is never taken from the request body."""
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise _unauthorized()
    token = decode_access_token(credentials.credentials, settings.secret_key)
    if token is None:
        raise _unauthorized()
    user = db.get(User, token.user_id)
    if user is None:
        raise _unauthorized()
    return user
