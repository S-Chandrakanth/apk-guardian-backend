"""Tests for the optional Mistral 7B LLM explanation service and its safe fallback."""

from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch
import pytest

from app.config import Settings
from app.services.llm_explainer import explain_findings


def make_dummy_settings(enable_ai=False, api_key=None):
    return Settings(
        secret_key="a" * 32,
        enable_ai_explanations=enable_ai,
        mistral_api_key=api_key,
        mistral_model="open-mistral-7b",
    )


def test_explainer_disabled_returns_rule_based():
    settings = make_dummy_settings(enable_ai=False, api_key="test-key")
    text, src = explain_findings(
        rule_based_explanation="Original rule-based text.",
        findings=[{"label": "OBSERVED", "text": "Package: com.app"}],
        classification="BENIGN",
        risk_score=10,
        risk_level="LOW",
        settings=settings,
    )
    assert text == "Original rule-based text."
    assert src == "rule_based"


def test_explainer_no_key_returns_rule_based():
    settings = make_dummy_settings(enable_ai=True, api_key=None)
    text, src = explain_findings(
        rule_based_explanation="Original rule-based text.",
        findings=[{"label": "OBSERVED", "text": "Package: com.app"}],
        classification="BENIGN",
        risk_score=10,
        risk_level="LOW",
        settings=settings,
    )
    assert text == "Original rule-based text."
    assert src == "rule_based"


def test_explainer_network_failure_falls_back_safely():
    settings = make_dummy_settings(enable_ai=True, api_key="valid-looking-key")
    with patch("urllib.request.urlopen", side_effect=OSError("Network unreachable")):
        text, src = explain_findings(
            rule_based_explanation="Original rule-based text.",
            findings=[{"label": "OBSERVED", "text": "Package: com.app"}],
            classification="SUSPICIOUS",
            risk_score=60,
            risk_level="HIGH",
            settings=settings,
        )
        assert text == "Original rule-based text."
        assert src == "rule_based"


def test_explainer_success_returns_ai_source():
    settings = make_dummy_settings(enable_ai=True, api_key="valid-key")
    mock_response_body = json.dumps({
        "choices": [
            {
                "message": {
                    "content": "This application requests sensitive SMS permissions without legitimate messaging features."
                }
            }
        ]
    }).encode("utf-8")

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = mock_response_body
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        text, src = explain_findings(
            rule_based_explanation="Original rule-based text.",
            findings=[{"label": "OBSERVED", "text": "Package: com.app"}],
            classification="MALICIOUS",
            risk_score=85,
            risk_level="CRITICAL",
            settings=settings,
        )
        assert src == "ai"
        assert "sensitive SMS permissions" in text
