"""Rule-based report text. Deterministic, built only from recorded evidence.

This is NOT the AI explanation component (a later phase). It never calls a model, never
changes a classification, and says "rule_based" as its source so clients can tell.
Each finding carries an evidence label:
  OBSERVED   read directly from the APK
  ML         output of the trained classifier
  HEURISTIC  a hand-written rule in risk.py
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from app.services.risk import Assessment

LIMITATIONS = [
    "This is a static analysis prototype. The APK was never run, so runtime behaviour is unknown.",
    "Packed, obfuscated, or dynamically loaded code can hide behaviour from static analysis.",
    "Counts of method references show what the code could call, not what it does at runtime.",
    "Requesting a permission or referencing an API is not proof of malicious intent; many legitimate apps do both.",
    "Model scores are not calibrated probabilities of infection, and this application does not establish how accurate the model is on real-world APKs.",
    "The risk score and level come from hand-written draft rules that have not been validated on real data.",
    "Absence of indicators does not prove an APK is safe.",
]

RECOMMENDATIONS = {
    "MALICIOUS": [
        "Do not install or open this file.",
        "If it is already installed, uninstall it and run a scan with Google Play Protect or a reputable mobile security app.",
        "Change passwords for accounts used on the device if you opened the app, and watch for unexpected SMS or charges.",
        "Confirm the result with another scanner before drawing a final conclusion.",
    ],
    "SUSPICIOUS": [
        "Install only if you obtained the file from the developer's official site or a trusted store.",
        "Compare the requested permissions with what the app is supposed to do; decline the app if they do not match.",
        "Confirm with another scanner and have someone with security experience review it if the app matters.",
    ],
    "BENIGN": [
        "No strong indicators were found by this prototype, which does not guarantee the app is safe.",
        "Prefer apps from official stores and keep Google Play Protect enabled.",
    ],
    "NEEDS REVIEW": [
        "Treat this APK as unverified: this prototype could not reach a confident conclusion.",
        "Do not install it unless you trust its source; scan it with additional tools or ask someone with security experience.",
    ],
}


def _model_note(model: Mapping[str, Any]) -> Optional[str]:
    if model.get("trained_on_synthetic_test_data"):
        return "WARNING: this model was trained on synthetic test data; its output is meaningless."
    return None


def build_report(
    package_name: Optional[str],
    features: Mapping[str, Optional[int]],
    indicators: Mapping[str, Any],
    assessment: Assessment,
    prediction: Optional[Mapping[str, Any]],
    model: Mapping[str, Any],
    extraction_status: str,
    error_code: Optional[str],
    settings: Optional[Any] = None,
) -> dict[str, Any]:
    findings: list[dict[str, str]] = []

    if extraction_status == "FAILED":
        findings.append({"label": "OBSERVED", "text": f"The file could not be analysed (reason code: {error_code})."})
    else:
        findings.append({"label": "OBSERVED", "text": f"Package name: {package_name}."})
        counts = (
            f"Manifest declares {features['permissions_count']} permission(s), {features['activities_count']} "
            f"activit(ies), {features['services_count']} service(s) and {features['receivers_count']} receiver(s)."
        )
        findings.append({"label": "OBSERVED", "text": counts})
        if features.get("strings_count") is not None:
            findings.append({
                "label": "OBSERVED",
                "text": f"The DEX code contains {features['strings_count']} unique strings and "
                        f"{features['api_calls_count']} distinct external method references (static references, not runtime calls).",
            })
        elif extraction_status == "PARTIAL":
            findings.append({"label": "OBSERVED", "text": "The DEX code could not be read, so code-level features are unavailable."})

    if prediction is not None:
        findings.append({
            "label": "ML",
            "text": f"The model gave a malware score of {prediction['malicious_probability']:.3f} "
                    f"(benign score {prediction['benign_probability']:.3f}). These are model scores, not certainty.",
        })
        note = _model_note(model)
        if note:
            findings.append({"label": "ML", "text": note})
    else:
        findings.append({"label": "ML", "text": "No model prediction was made for this scan."})

    for component in assessment.components:
        if component.label == "HEURISTIC" and component.points:
            for item in component.evidence:
                findings.append({"label": "HEURISTIC", "text": f"{item} (+{component.points:g} of {component.max_points:g} possible points)."})

    if assessment.score is None:
        headline = "No assessment could be made because the file could not be analysed."
    else:
        headline = (
            f"Overall result: {assessment.classification}. Heuristic risk level: {assessment.level} "
            f"(score {assessment.score}/100, basis {assessment.basis})."
        )
    explanation = " ".join([headline, *assessment.reasons,
                            "This text was generated by fixed rules from the evidence listed in the findings."])
    explanation_source = "rule_based"

    if settings is not None:
        try:
            from app.services.llm_explainer import explain_findings
            explanation, explanation_source = explain_findings(
                rule_based_explanation=explanation,
                findings=findings,
                classification=assessment.classification,
                risk_score=assessment.score,
                risk_level=assessment.level,
                settings=settings,
            )
        except Exception:
            pass

    return {
        "explanation": explanation,
        "explanation_source": explanation_source,
        "findings": findings,
        "recommendations": list(RECOMMENDATIONS[assessment.classification]),
        "limitations": list(LIMITATIONS),
    }
