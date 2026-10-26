"""SHA-256 helpers. Files are hashed in chunks so large APKs never load fully into memory."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

CHUNK_SIZE = 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = CHUNK_SIZE) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_valid_sha256(value: object) -> bool:
    """True for a 64-character lowercase hexadecimal string."""
    return isinstance(value, str) and _SHA256_RE.match(value) is not None


def normalize_sha256(value: object) -> str | None:
    """Return the lowercase, stripped hash if it is valid, otherwise None."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip().lower()
    return cleaned if _SHA256_RE.match(cleaned) else None
