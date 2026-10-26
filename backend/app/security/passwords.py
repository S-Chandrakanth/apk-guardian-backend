"""Password hashing with scrypt from Python's standard library (no native dependency).

Stored format:  scrypt$<N>$<r>$<p>$<salt b64>$<hash b64>
The cost parameters live inside each hash, so they can be raised later and old hashes
still verify. Parameters follow the OWASP guidance for scrypt (N=2^15, r=8, p=3).
If the Python build has no scrypt, PBKDF2-HMAC-SHA256 with 600,000 iterations is used.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets

SCRYPT_N = 2 ** 15
SCRYPT_R = 8
SCRYPT_P = 3
PBKDF2_ITERATIONS = 600_000
SALT_BYTES = 16
KEY_BYTES = 32
MAX_PASSWORD_BYTES = 1024  # refuse absurd inputs so hashing cannot be used to burn CPU

_HAS_SCRYPT = hasattr(hashlib, "scrypt")


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"), validate=True)


def _scrypt(password: bytes, salt: bytes, n: int, r: int, p: int) -> bytes:
    return hashlib.scrypt(password, salt=salt, n=n, r=r, p=p, dklen=KEY_BYTES, maxmem=256 * 1024 * 1024)


def hash_password(password: str) -> str:
    raw = password.encode("utf-8")
    if len(raw) > MAX_PASSWORD_BYTES:
        raise ValueError("password too long")
    salt = secrets.token_bytes(SALT_BYTES)
    if _HAS_SCRYPT:
        digest = _scrypt(raw, salt, SCRYPT_N, SCRYPT_R, SCRYPT_P)
        return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${_b64(salt)}${_b64(digest)}"
    digest = hashlib.pbkdf2_hmac("sha256", raw, salt, PBKDF2_ITERATIONS, dklen=KEY_BYTES)
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, stored: str) -> bool:
    """Constant-time check. Returns False (never raises) for malformed hashes or passwords."""
    try:
        raw = password.encode("utf-8")
        if len(raw) > MAX_PASSWORD_BYTES:
            return False
        parts = stored.split("$")
        if parts[0] == "scrypt" and len(parts) == 6 and _HAS_SCRYPT:
            n, r, p = int(parts[1]), int(parts[2]), int(parts[3])
            expected = _unb64(parts[5])
            actual = _scrypt(raw, _unb64(parts[4]), n, r, p)
        elif parts[0] == "pbkdf2_sha256" and len(parts) == 4:
            expected = _unb64(parts[3])
            actual = hashlib.pbkdf2_hmac("sha256", raw, _unb64(parts[2]), int(parts[1]), dklen=len(expected))
        else:
            return False
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError, OverflowError, MemoryError):
        return False


def needs_rehash(stored: str) -> bool:
    parts = stored.split("$")
    if parts[0] == "scrypt" and len(parts) == 6:
        return (int(parts[1]), int(parts[2]), int(parts[3])) != (SCRYPT_N, SCRYPT_R, SCRYPT_P)
    return True


# A real hash computed once, used to keep login timing similar when the email is unknown.
_DUMMY_HASH: str | None = None


def dummy_verify(password: str) -> None:
    """Spend roughly the same time as a real verification (reduces user enumeration by timing)."""
    global _DUMMY_HASH
    if _DUMMY_HASH is None:
        _DUMMY_HASH = hash_password(secrets.token_urlsafe(16))
    verify_password(password, _DUMMY_HASH)
