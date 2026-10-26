"""Loader for the pinned permission and API lists used by the feature extractor."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from app.ml.schema import EXTRACTOR_BASE_VERSION

_RESOURCE_DIR = Path(__file__).resolve().parent / "resources"


@lru_cache(maxsize=None)
def _load(name: str) -> dict:
    with open(_RESOURCE_DIR / name, "r", encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def dangerous_permissions() -> frozenset[str]:
    return frozenset(_load("dangerous_permissions.json")["permissions"])


@lru_cache(maxsize=1)
def suspicious_apis() -> dict[tuple[str, str], str]:
    """Map (class descriptor, method name) -> reason."""
    out: dict[tuple[str, str], str] = {}
    for entry in _load("suspicious_apis.json")["apis"]:
        for method in entry["methods"]:
            out[(entry["class"], method)] = entry["reason"]
    return out


def permission_list_version() -> str:
    return _load("dangerous_permissions.json")["version"]


def api_list_version() -> str:
    return _load("suspicious_apis.json")["version"]


def extractor_version() -> str:
    """Full feature-definition version: code version plus both pinned-list versions.

    A trained model records this string. If any list or the code changes, the string
    changes, and a model trained under a different version must not be used silently.
    """
    return f"{EXTRACTOR_BASE_VERSION}+perm{permission_list_version()}+api{api_list_version()}"
