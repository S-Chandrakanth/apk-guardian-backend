"""Optional Mistral 7B LLM-based explanation service with safe fallback.

If the Mistral API is not configured, unreachable, or fails, the report seamlessly
falls back to the deterministic, rule-based explanation without affecting the scan.
The findings and classification verdicts are never altered by the LLM.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Mapping, Optional
import urllib.request
import urllib.error

from app.config import Settings

logger = logging.getLogger("apkguardian.llm")


def explain_findings(
    rule_based_explanation: str,
    findings: list[dict[str, str]],
    classification: str,
    risk_score: Optional[int],
    risk_level: Optional[str],
    settings: Settings,
) -> tuple[str, str]:
    """Generates an explanation using Mistral 7B if enabled and configured;
    otherwise falls back safely to rule-based explanation.
    
    Returns:
        tuple[str, str]: (explanation_text, explanation_source)
        where explanation_source is either 'ai' or 'rule_based'.
    """
    if not settings.enable_ai_explanations or not settings.mistral_api_key:
        return rule_based_explanation, "rule_based"

    try:
        findings_summary = "\n".join(f"- [{f['label']}] {f['text']}" for f in findings[:8])
        prompt = (
            f"Classification Verdict: {classification}\n"
            f"Heuristic Risk Score: {risk_score if risk_score is not None else 'N/A'}/100 ({risk_level or 'Unknown'})\n"
            f"Key Findings from Static Analysis:\n{findings_summary}\n\n"
            "Explain in 2-3 concise, professional sentences why this Android APK is assessed this way, "
            "highlighting specific permissions or indicators found. Keep the tone factual and explainable."
        )

        payload = {
            "model": settings.mistral_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an explainable cybersecurity assistant in APK Guardian. "
                        "Provide a concise, objective summary of the static analysis evidence. "
                        "Never claim the app ran dynamically or speculate beyond the provided facts."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 150,
            "temperature": 0.2,
        }

        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            settings.mistral_api_url,
            data=req_data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {settings.mistral_api_key}",
                "User-Agent": "APKGuardian-Backend/1.0",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=8.0) as resp:
            if resp.status == 200:
                body = json.loads(resp.read().decode("utf-8"))
                choice = body.get("choices", [{}])[0]
                text = choice.get("message", {}).get("content", "").strip()
                if text:
                    logger.info("Successfully generated AI explanation via Mistral 7B.")
                    return text, "ai"

    except Exception as exc:
        logger.warning("Mistral 7B explanation unavailable (%s); using rule-based fallback.", exc)

    return rule_based_explanation, "rule_based"
