"""Signed access tokens (JWT, HS256). The algorithm is pinned on decode."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Optional

import jwt

ALGORITHM = "HS256"
ISSUER = "apk-guardian"


@dataclass(frozen=True)
class TokenData:
    user_id: int
    expires_at: int


def create_access_token(user_id: int, secret_key: str, expires_minutes: int) -> tuple[str, int]:
    """Return (token, lifetime in seconds)."""
    now = int(time.time())
    lifetime = expires_minutes * 60
    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": now + lifetime,
        "iss": ISSUER,
        "jti": uuid.uuid4().hex,
    }
    return jwt.encode(payload, secret_key, algorithm=ALGORITHM), lifetime


def decode_access_token(token: str, secret_key: str) -> Optional[TokenData]:
    """Return the token's data, or None for any invalid, expired, or tampered token."""
    try:
        claims = jwt.decode(
            token,
            secret_key,
            algorithms=[ALGORITHM],
            issuer=ISSUER,
            options={"require": ["exp", "iat", "sub", "iss"]},
        )
        return TokenData(user_id=int(claims["sub"]), expires_at=int(claims["exp"]))
    except (jwt.PyJWTError, ValueError, TypeError, KeyError):
        return None
