"""Shared test fixtures. Every APK used in tests is built by tests/fixtures/apk_builder.py:
tiny, benign, synthetic, unsigned, and containing no executable code."""

from __future__ import annotations

import hashlib

import pytest

from tests.fixtures.apk_builder import build_apk


@pytest.fixture
def make_apk(tmp_path):
    counter = {"n": 0}

    def _make(name: str | None = None, **kwargs):
        counter["n"] += 1
        return build_apk(tmp_path / (name or f"sample{counter['n']}.apk"), **kwargs)

    return _make


@pytest.fixture
def valid_apk(make_apk):
    return make_apk("valid.apk")


def sha(text: str) -> str:
    """Deterministic fake-but-well-formed SHA-256 string for dataset-table tests."""
    return hashlib.sha256(text.encode()).hexdigest()
