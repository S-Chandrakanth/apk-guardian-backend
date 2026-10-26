from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db, get_settings
from app.config import Settings
from app.database.models import User
from app.schemas.auth import LoginRequest, RegisterRequest, TokenOut, UserOut
from app.security.passwords import dummy_verify, hash_password, needs_rehash, verify_password
from app.security.tokens import create_access_token

logger = logging.getLogger("apkguardian.auth")
router = APIRouter(tags=["auth"])

EMAIL_TAKEN = "An account with this email already exists."


@router.post("/register", response_model=UserOut, status_code=201, summary="Create an account")
def register(body: RegisterRequest, db: Session = Depends(get_db)) -> User:
    if db.scalar(select(User.id).where(User.email == body.email)) is not None:
        raise HTTPException(status_code=409, detail=EMAIL_TAKEN)
    user = User(name=body.name, email=body.email, password_hash=hash_password(body.password))
    try:
        db.add(user)
        db.commit()
    except IntegrityError:
        db.rollback()
        if db.scalar(select(User.id).where(User.email == body.email)) is not None:  # lost a race
            raise HTTPException(status_code=409, detail=EMAIL_TAKEN)
        raise
    return user


@router.post("/login", response_model=TokenOut, summary="Log in and receive an access token")
def login(
    body: LoginRequest,
    request: Request,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> TokenOut:
    limiter = request.app.state.login_limiter
    client = request.client.host if request.client else "unknown"
    key = f"{client}|{body.email}"

    wait = limiter.retry_after(key)
    if wait:
        raise HTTPException(
            status_code=429,
            detail="Too many failed login attempts. Try again later.",
            headers={"Retry-After": str(wait)},
        )

    user = db.scalar(select(User).where(User.email == body.email))
    if user is None:
        dummy_verify(body.password)  # similar timing whether or not the account exists
        valid = False
    else:
        valid = verify_password(body.password, user.password_hash)

    if not valid or user is None:
        limiter.record_failure(key)
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    limiter.reset(key)
    if needs_rehash(user.password_hash):  # upgrade old hashes to current cost settings
        try:
            user.password_hash = hash_password(body.password)
            db.commit()
        except Exception:
            db.rollback()
            logger.warning("password rehash failed", exc_info=True)

    token, lifetime = create_access_token(user.id, settings.secret_key, settings.access_token_expire_minutes)
    return TokenOut(access_token=token, expires_in=lifetime)


@router.get("/me", response_model=UserOut, summary="The logged-in user")
def me(user: User = Depends(get_current_user)) -> User:
    return user
