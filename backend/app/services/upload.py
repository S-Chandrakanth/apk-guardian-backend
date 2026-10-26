"""Safe handling of uploaded files.

* The client's file name is only ever used as a sanitised DISPLAY name. The file on disk gets
  a random name inside a controlled directory, so a hostile name cannot reach the filesystem.
* The upload is streamed to disk in chunks with a hard size cap and hashed on the way.
* Nothing is executed, imported, or unpacked here.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Optional

logger = logging.getLogger("apkguardian.upload")

CHUNK = 1024 * 1024
ZIP_MAGIC = b"PK\x03\x04"
MAX_DISPLAY_NAME = 100
TMP_PREFIX = "scan_"


class UploadRejected(Exception):
    """The upload cannot be accepted. Carries the HTTP status and a client-safe message."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass(frozen=True)
class SavedUpload:
    path: Path
    size: int
    sha256: str


def has_apk_extension(raw_name: Optional[str]) -> bool:
    base = (raw_name or "").replace("\\", "/").rsplit("/", 1)[-1]
    return base.lower().endswith(".apk") and len(base) > 4


def sanitize_filename(raw_name: Optional[str]) -> str:
    """Return a safe display name: no path parts, no control characters, bounded length."""
    base = (raw_name or "").replace("\\", "/").rsplit("/", 1)[-1]
    base = unicodedata.normalize("NFC", base)
    cleaned = "".join(ch if (ch.isalnum() or ch in "._()- ") else "_" for ch in base)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    stem = cleaned[:-4] if cleaned.lower().endswith(".apk") else cleaned
    stem = stem.strip(" .")[: MAX_DISPLAY_NAME - 4]
    return f"{stem}.apk" if stem else "upload.apk"


def save_upload(
    stream: BinaryIO,
    tmp_dir: Path,
    max_bytes: int,
    min_bytes: int,
) -> SavedUpload:
    """Stream `stream` into a new file in tmp_dir. Raises UploadRejected for bad content.

    On any failure the partial file is deleted before the exception leaves this function.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=TMP_PREFIX, suffix=".apk", dir=tmp_dir)
    path = Path(name)
    digest = hashlib.sha256()
    size = 0
    try:
        with os.fdopen(fd, "wb") as out:
            first = True
            while True:
                chunk = stream.read(CHUNK)
                if not chunk:
                    break
                if first:
                    first = False
                    if not chunk.startswith(ZIP_MAGIC):
                        raise UploadRejected(400, "The file is not an APK (it is not a ZIP archive).")
                size += len(chunk)
                if size > max_bytes:
                    raise UploadRejected(413, f"The file is larger than the {max_bytes // (1024 * 1024)} MB limit.")
                digest.update(chunk)
                out.write(chunk)
        if size == 0:
            raise UploadRejected(400, "The uploaded file is empty.")
        if size < min_bytes:
            raise UploadRejected(400, "The file is too small to be a valid APK.")
        return SavedUpload(path=path, size=size, sha256=digest.hexdigest())
    except BaseException:
        remove_quietly(path)
        raise


def remove_quietly(path: Path, attempts: int = 5) -> None:
    """Delete a temporary file. Retries briefly because Windows may still hold a lock."""
    for attempt in range(attempts):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            time.sleep(0.1 * (attempt + 1))
        except OSError:
            logger.warning("could not delete temporary upload file")
            return
    logger.warning("temporary upload file is still locked and was not deleted")


def cleanup_stale_uploads(tmp_dir: Path, max_age_seconds: int = 3600) -> int:
    """Delete leftover temporary uploads (for example after a crash). Returns the number removed."""
    removed = 0
    if not tmp_dir.is_dir():
        return removed
    cutoff = time.time() - max_age_seconds
    for entry in tmp_dir.glob(f"{TMP_PREFIX}*.apk"):
        try:
            if entry.is_file() and entry.stat().st_mtime < cutoff:
                entry.unlink()
                removed += 1
        except OSError:
            continue
    return removed
