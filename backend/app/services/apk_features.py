"""Static APK feature extraction using Androguard.

Safety rules enforced here:
* APKs are only PARSED. They are never installed or executed.
* Every failure is reported with a status and an error code. A feature that could not be
  read is None. It is never replaced with 0 or any other invented value.
* Heavy parsing can be run in a separate worker process with a hard timeout
  (see IsolatedExtractor) so a malformed file cannot hang the caller.

Statuses
    OK       all eight features were read.
    PARTIAL  the manifest was read but the DEX files could not be (or the reverse is
             impossible, because without a manifest there is nothing to report): DEX
             features are None.
    FAILED   the file could not be analysed; all features are None.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import struct
import zipfile
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from app.ml import feature_lists
from app.ml.schema import (
    DEX_FEATURES,
    FEATURE_NAMES,
    STATUS_FAILED,
    STATUS_OK,
    STATUS_PARTIAL,
)
from app.services.hashing import sha256_file

logger = logging.getLogger(__name__)

ANDROID_NS = "http://schemas.android.com/apk/res/android"

# Error codes
FILE_NOT_FOUND = "FILE_NOT_FOUND"
FILE_TOO_SMALL = "FILE_TOO_SMALL"
FILE_TOO_LARGE = "FILE_TOO_LARGE"
NOT_A_ZIP = "NOT_A_ZIP"
CORRUPT_ZIP = "CORRUPT_ZIP"
ZIP_LIMITS_EXCEEDED = "ZIP_LIMITS_EXCEEDED"
MISSING_MANIFEST = "MISSING_MANIFEST"
MISSING_DEX = "MISSING_DEX"
MANIFEST_PARSE_ERROR = "MANIFEST_PARSE_ERROR"
DEX_PARSE_ERROR = "DEX_PARSE_ERROR"
TIMEOUT = "TIMEOUT"
WORKER_CRASH = "WORKER_CRASH"
UNEXPECTED_ERROR = "UNEXPECTED_ERROR"


@dataclass(frozen=True)
class ExtractionLimits:
    min_size_bytes: int = 512
    max_size_bytes: int = 100 * 1024 * 1024
    max_zip_entries: int = 50_000
    max_total_uncompressed_bytes: int = 2 * 1024 ** 3
    max_manifest_bytes: int = 20 * 1024 ** 2
    max_dex_bytes: int = 200 * 1024 ** 2


@dataclass
class ExtractionResult:
    status: str
    features: dict[str, Optional[int]] = field(
        default_factory=lambda: {name: None for name in FEATURE_NAMES}
    )
    package_name: Optional[str] = None
    sha256: Optional[str] = None
    file_size: Optional[int] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    indicators: dict[str, Any] = field(default_factory=dict)
    extractor_version: str = field(default_factory=feature_lists.extractor_version)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExtractionResult":
        return cls(**data)

    @property
    def is_complete(self) -> bool:
        return self.status == STATUS_OK


def _failed(code: str, message: str, **kwargs: Any) -> ExtractionResult:
    return ExtractionResult(status=STATUS_FAILED, error_code=code, error_message=message, **kwargs)


def _quiet_androguard() -> None:
    """Androguard logs very verbosely (loguru in 4.x, logging in 3.x); silence both."""
    try:
        from loguru import logger as loguru_logger

        loguru_logger.disable("androguard")
        loguru_logger.remove()
    except Exception:  # pragma: no cover - loguru is not installed with every version
        pass
    logging.getLogger("androguard").setLevel(logging.CRITICAL)


class ExtractorEnvironmentError(RuntimeError):
    """Androguard is missing or has an API this code does not support.

    This is a problem with the installation, not with the APK being analysed, so it is
    raised instead of being recorded as a FAILED sample.
    """


_ANDROGUARD: Optional[tuple[Any, Any, str]] = None


def load_androguard() -> tuple[Any, Any, str]:
    """Return (APK class, DEX class, version string) for the installed Androguard.

    Supported: 4.x (androguard.core.apk / androguard.core.dex) and 3.x
    (androguard.core.bytecodes.apk / androguard.core.bytecodes.dvm).
    """
    global _ANDROGUARD
    if _ANDROGUARD is not None:
        return _ANDROGUARD
    _quiet_androguard()
    try:
        from androguard.core.apk import APK
        from androguard.core.dex import DEX
    except ImportError:
        try:
            from androguard.core.bytecodes.apk import APK
            from androguard.core.bytecodes.dvm import DalvikVMFormat as DEX
        except ImportError as exc:
            raise ExtractorEnvironmentError(
                "Androguard is not installed, or its version is not supported. "
                "Run: pip install -r backend/requirements.txt  "
                f"(import error: {exc})"
            ) from exc
    try:
        import androguard

        version = str(getattr(androguard, "__version__", "unknown"))
    except ImportError:  # pragma: no cover
        version = "unknown"
    _ANDROGUARD = (APK, DEX, version)
    return _ANDROGUARD


def androguard_version() -> str:
    return load_androguard()[2]


# --------------------------------------------------------------------------
# Pre-checks (cheap, no Androguard)
# --------------------------------------------------------------------------
def _precheck(path: Path, limits: ExtractionLimits) -> Optional[ExtractionResult]:
    """Return a FAILED result if the file is unusable, otherwise None."""
    try:
        size = path.stat().st_size
    except OSError:
        return _failed(FILE_NOT_FOUND, "File does not exist or cannot be read.")
    if not path.is_file():
        return _failed(FILE_NOT_FOUND, "Path is not a regular file.", file_size=size)
    if size < limits.min_size_bytes:
        return _failed(
            FILE_TOO_SMALL,
            f"File is {size} bytes; the minimum accepted size is {limits.min_size_bytes} bytes.",
            file_size=size,
        )
    if size > limits.max_size_bytes:
        return _failed(
            FILE_TOO_LARGE,
            f"File is {size} bytes; the maximum accepted size is {limits.max_size_bytes} bytes.",
            file_size=size,
        )
    with open(path, "rb") as fh:
        magic = fh.read(4)
    if magic != b"PK\x03\x04":
        return _failed(NOT_A_ZIP, "File does not start with a ZIP/APK signature.", file_size=size)

    try:
        with zipfile.ZipFile(path) as zf:
            infos = zf.infolist()
            if len(infos) > limits.max_zip_entries:
                return _failed(
                    ZIP_LIMITS_EXCEEDED,
                    f"Archive has {len(infos)} entries; limit is {limits.max_zip_entries}.",
                    file_size=size,
                )
            total = sum(i.file_size for i in infos)
            if total > limits.max_total_uncompressed_bytes:
                return _failed(
                    ZIP_LIMITS_EXCEEDED,
                    f"Archive would expand to {total} bytes; limit is "
                    f"{limits.max_total_uncompressed_bytes}.",
                    file_size=size,
                )
            names = {i.filename: i for i in infos}
            manifest = names.get("AndroidManifest.xml")
            if manifest is None:
                return _failed(
                    MISSING_MANIFEST, "Archive has no AndroidManifest.xml.", file_size=size
                )
            if manifest.file_size > limits.max_manifest_bytes:
                return _failed(
                    ZIP_LIMITS_EXCEEDED, "AndroidManifest.xml is unreasonably large.", file_size=size
                )
            for info in infos:
                if info.filename.endswith(".dex") and "/" not in info.filename:
                    if info.file_size > limits.max_dex_bytes:
                        return _failed(
                            ZIP_LIMITS_EXCEEDED,
                            f"{info.filename} is unreasonably large.",
                            file_size=size,
                        )
            # Reading the manifest verifies its CRC.
            zf.read(manifest)
    except zipfile.BadZipFile as exc:
        return _failed(CORRUPT_ZIP, f"Archive is corrupted: {exc}", file_size=size)
    except (OSError, EOFError, zipfile.LargeZipFile, NotImplementedError, RuntimeError) as exc:
        return _failed(CORRUPT_ZIP, f"Archive could not be read: {exc}", file_size=size)
    return None


# --------------------------------------------------------------------------
# Manifest and DEX readers
# --------------------------------------------------------------------------
def _qualify(package: str, name: str) -> str:
    """Turn a manifest component name into a fully qualified class name."""
    name = name.strip()
    if name.startswith("."):
        return package + name
    if "." not in name:
        return package + "." + name
    return name


def _manifest_root(apk: Any) -> Any:
    root = apk.get_android_manifest_xml()
    if root is None:
        raise ValueError("the manifest XML could not be decoded")
    return root


def _attribute_values(root: Any, tag: str, package: Optional[str]) -> set[str]:
    """Unique android:name values of every <tag> element, qualified when package is given.

    Components are read straight from the manifest XML (not from Androguard's helper methods)
    so the result is identical across Androguard versions. Androguard 3.x, for example,
    turns ".Svc" into "pkg..Svc".
    """
    attr = f"{{{ANDROID_NS}}}name"
    names: set[str] = set()
    for node in root.iter(tag):
        value = node.get(attr)
        if value and value.strip():
            names.add(_qualify(package, value) if package else value.strip())
    return names


def _read_manifest(apk: Any) -> tuple[str, dict[str, int], dict[str, list[str]]]:
    """Return (package, manifest feature values, sorted name lists for the report)."""
    root = _manifest_root(apk)
    package = (root.get("package") or "").strip()
    if not package:
        raise ValueError("manifest has no package attribute")
    permissions = _attribute_values(root, "uses-permission", None) | _attribute_values(
        root, "uses-permission-sdk-23", None
    )
    activities = _attribute_values(root, "activity", package) | _attribute_values(
        root, "activity-alias", package
    )
    services = _attribute_values(root, "service", package)
    receivers = _attribute_values(root, "receiver", package)
    dangerous = sorted(p for p in permissions if p in feature_lists.dangerous_permissions())
    values = {
        "permissions_count": len(permissions),
        "activities_count": len(activities),
        "services_count": len(services),
        "receivers_count": len(receivers),
        "dangerous_permissions": len(dangerous),
    }
    names = {
        "permissions": sorted(permissions),
        "dangerous_permission_names": dangerous,
        "activities": sorted(activities),
        "services": sorted(services),
        "receivers": sorted(receivers),
    }
    return package, values, names


def validate_dex_blob(blob: bytes) -> None:
    """Check a DEX file's own header, declared size and Adler-32 checksum.

    Done here, independent of Androguard, because some Androguard versions parse damaged
    DEX files without any error and would then yield wrong feature values.
    A DEX that fails this check is reported as unusable (PARTIAL), never guessed at.
    """
    if len(blob) < 0x70 or not blob.startswith(b"dex\n"):
        raise ValueError("classes.dex does not start with the DEX magic bytes")
    checksum, = struct.unpack_from("<I", blob, 8)
    declared_size, header_size = struct.unpack_from("<II", blob, 32)
    if header_size != 0x70:
        raise ValueError(f"DEX header size is {header_size}, expected 112")
    if declared_size != len(blob):
        raise ValueError(f"DEX header declares {declared_size} bytes but the file has {len(blob)}")
    if zlib.adler32(blob[12:]) & 0xFFFFFFFF != checksum:
        raise ValueError("DEX checksum mismatch (file is damaged or has been altered)")


def _read_dex(blobs: list[bytes]) -> tuple[dict[str, int], list[str]]:
    """Return (DEX feature values, sorted suspicious API matches like 'Class.method')."""
    _, DEX, _ = load_androguard()

    strings: set[str] = set()
    defined_classes: set[str] = set()
    method_refs: set[tuple[str, str, str]] = set()
    for blob in blobs:
        validate_dex_blob(blob)
        dex = DEX(blob)
        strings.update(dex.get_strings())
        defined_classes.update(dex.get_classes_names())
        ids = dex.get_methods_id_item()  # None when the DEX has no method references at all
        if ids is not None:
            for item in ids.get_obj():
                # Do not use get_triple(): it strips the leading "L" and trailing ";" from the class.
                proto = item.get_proto()
                descriptor = "".join(proto) if proto else ""
                method_refs.add((item.get_class_name(), item.get_name(), descriptor))

    external = {ref for ref in method_refs if ref[0] not in defined_classes}
    watch = feature_lists.suspicious_apis()
    matches = sorted({f"{c}.{m}" for c, m, _ in external if (c, m) in watch})
    values = {
        "api_calls_count": len(external),
        "strings_count": len(strings),
        "suspicious_api_calls": len(matches),
    }
    return values, matches


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def extract_features(
    path: str | Path, limits: Optional[ExtractionLimits] = None
) -> ExtractionResult:
    """Extract the eight static features from an APK.

    Never raises for a bad APK; it returns a FAILED or PARTIAL result instead. It does raise
    ExtractorEnvironmentError if Androguard itself is missing or unsupported, because that
    is an installation problem and must not be recorded as a property of the sample.
    """
    limits = limits or ExtractionLimits()
    path = Path(path)
    APK, _, ag_version = load_androguard()

    early = _precheck(path, limits)
    sha = None
    try:
        sha = sha256_file(path)
    except OSError:
        pass
    if early is not None:
        early.sha256 = sha
        return early

    size = path.stat().st_size
    try:
        apk = APK(str(path))
        if not apk.is_valid_APK():
            return _failed(
                MANIFEST_PARSE_ERROR,
                "Androguard could not validate this APK.",
                sha256=sha,
                file_size=size,
            )
        package, manifest_values, names = _read_manifest(apk)
    except zipfile.BadZipFile as exc:
        return _failed(CORRUPT_ZIP, f"Archive is corrupted: {exc}", sha256=sha, file_size=size)
    except Exception as exc:  # Androguard raises many exception types on malformed input
        return _failed(
            MANIFEST_PARSE_ERROR,
            f"Manifest could not be parsed: {type(exc).__name__}: {exc}",
            sha256=sha,
            file_size=size,
        )

    result = ExtractionResult(
        status=STATUS_OK,
        package_name=package,
        sha256=sha,
        file_size=size,
    )
    result.features.update(manifest_values)
    result.indicators.update(names)
    result.indicators["androguard_version"] = ag_version

    try:
        blobs = list(apk.get_all_dex())
    except Exception as exc:
        blobs = []
        result.status = STATUS_PARTIAL
        result.error_code = DEX_PARSE_ERROR
        result.error_message = f"DEX files could not be read: {type(exc).__name__}: {exc}"
        return result

    if not blobs:
        result.status = STATUS_PARTIAL
        result.error_code = MISSING_DEX
        result.error_message = "APK contains no classes.dex; DEX features are unavailable."
        return result

    try:
        dex_values, matches = _read_dex(blobs)
    except Exception as exc:
        result.status = STATUS_PARTIAL
        result.error_code = DEX_PARSE_ERROR
        result.error_message = f"DEX could not be parsed: {type(exc).__name__}: {exc}"
        for name in DEX_FEATURES:
            result.features[name] = None
        return result

    result.features.update(dex_values)
    result.indicators["suspicious_api_matches"] = matches
    result.indicators["dex_file_count"] = len(blobs)
    return result


# --------------------------------------------------------------------------
# Process isolation with a hard timeout
# --------------------------------------------------------------------------
def _worker_main(conn: Any, limits_dict: dict[str, Any]) -> None:
    limits = ExtractionLimits(**limits_dict)
    _quiet_androguard()
    while True:
        try:
            message = conn.recv()
        except (EOFError, OSError):
            break
        if message is None:
            break
        try:
            payload = extract_features(message, limits).to_dict()
        except ExtractorEnvironmentError as exc:
            payload = {"environment_error": str(exc)}  # installation problem, not a sample result
        except Exception as exc:  # extract_features should not raise; belt and braces
            payload = _failed(UNEXPECTED_ERROR, f"{type(exc).__name__}: {exc}").to_dict()
        try:
            conn.send(payload)
        except (BrokenPipeError, OSError):
            break


class IsolatedExtractor:
    """Runs extraction in a child process and kills it if it exceeds the timeout.

    One instance handles one APK at a time. Use one instance per thread for parallelism.
    On Windows the child is started with "spawn", so callers' scripts must use the usual
    ``if __name__ == "__main__":`` guard.
    """

    def __init__(
        self,
        limits: Optional[ExtractionLimits] = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.limits = limits or ExtractionLimits()
        self.timeout_seconds = timeout_seconds
        self._ctx = multiprocessing.get_context("spawn")
        self._process: Any = None
        self._conn: Any = None

    def _start(self) -> None:
        parent_conn, child_conn = self._ctx.Pipe()
        process = self._ctx.Process(
            target=_worker_main,
            args=(child_conn, asdict(self.limits)),
            daemon=True,
        )
        process.start()
        child_conn.close()
        self._process, self._conn = process, parent_conn

    def _stop(self) -> None:
        if self._process is not None:
            if self._process.is_alive():
                self._process.kill()
            self._process.join(timeout=5)
        if self._conn is not None:
            try:
                self._conn.close()
            except OSError:
                pass
        self._process = self._conn = None

    def extract(self, path: str | Path) -> ExtractionResult:
        path = Path(path)
        if self._process is None or not self._process.is_alive():
            self._stop()
            self._start()

        def fallback(code: str, message: str) -> ExtractionResult:
            sha = size = None
            try:
                sha, size = sha256_file(path), path.stat().st_size
            except OSError:
                pass
            return _failed(code, message, sha256=sha, file_size=size)

        try:
            self._conn.send(str(path))
            if not self._conn.poll(self.timeout_seconds):
                self._stop()
                return fallback(TIMEOUT, f"Analysis exceeded {self.timeout_seconds:g} seconds.")
            payload = self._conn.recv()
            if "environment_error" in payload:
                self._stop()
                raise ExtractorEnvironmentError(payload["environment_error"])
            return ExtractionResult.from_dict(payload)
        except (EOFError, BrokenPipeError, ConnectionResetError, OSError):
            self._stop()
            return fallback(WORKER_CRASH, "The analysis process ended unexpectedly.")

    def close(self) -> None:
        if self._conn is not None and self._process is not None and self._process.is_alive():
            try:
                self._conn.send(None)
            except (OSError, BrokenPipeError):
                pass
        self._stop()

    def __enter__(self) -> "IsolatedExtractor":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def extract_features_isolated(
    path: str | Path,
    limits: Optional[ExtractionLimits] = None,
    timeout_seconds: float = 120.0,
) -> ExtractionResult:
    """Convenience wrapper: one APK, one short-lived worker process, hard timeout."""
    with IsolatedExtractor(limits=limits, timeout_seconds=timeout_seconds) as extractor:
        return extractor.extract(path)
