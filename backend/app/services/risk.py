"""Risk assessment and final classification (DRAFT, heuristic, not validated on real data).

Every number here is a documented, hand-chosen rule. None of it is a calibrated probability
of infection, and the thresholds have not been tuned or validated against real APKs.

Score (0-100) = ML points + heuristic points
  ML points (0-50)         50 x model malware probability, only when a model produced one
  Permission points (0-20) dangerous permissions (1 each, max 10) + combination bonuses
  API points (0-20)        4 per distinct suspicious API reference, max 5 counted
  Structural points (0-10) two simple indicators, 5 each

Levels: LOW 0-24, MEDIUM 25-49, HIGH 50-74, CRITICAL 75-100.
Without a model the score can reach at most 50, so heuristics alone never exceed HIGH.

Classification
  NEEDS REVIEW  extraction failed or partial; no usable model score; score in the
                uncertainty band; or a low model score contradicts strong heuristic evidence
  MALICIOUS     score >= 0.80 AND at least one supporting heuristic indicator
  SUSPICIOUS    score above the band but MALICIOUS conditions not met
  BENIGN        score below the band AND no strong heuristic evidence
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

RISK_LOGIC_VERSION = "1.0.0-draft"

REVIEW_BAND = (0.40, 0.60)
MALICIOUS_THRESHOLD = 0.80
STRONG_HEURISTIC_POINTS = 20.0
LEVEL_BOUNDS = ((75, "CRITICAL"), (50, "HIGH"), (25, "MEDIUM"), (0, "LOW"))

PERMISSION = "android.permission."
SMS_SEND, SMS_READ, SMS_RECV = PERMISSION + "SEND_SMS", PERMISSION + "READ_SMS", PERMISSION + "RECEIVE_SMS"
CONTACTS, CALL_LOG = PERMISSION + "READ_CONTACTS", PERMISSION + "READ_CALL_LOG"
AUDIO, CAMERA, INTERNET = PERMISSION + "RECORD_AUDIO", PERMISSION + "CAMERA", PERMISSION + "INTERNET"

DYNAMIC_LOADERS = {
    "Ldalvik/system/DexClassLoader;",
    "Ldalvik/system/PathClassLoader;",
    "Ldalvik/system/InMemoryDexClassLoader;",
}
REFLECTION_CALL = "Ljava/lang/reflect/Method;.invoke"


@dataclass
class Component:
    name: str
    label: str  # "ML" or "HEURISTIC"
    points: Optional[float]  # None = could not be computed (evidence unavailable)
    max_points: float
    evidence: list[str] = field(default_factory=list)


@dataclass
class Assessment:
    classification: str
    reasons: list[str]
    score: Optional[int]
    level: Optional[str]
    basis: Optional[str]
    components: list[Component]
    heuristic_points: Optional[float]


def level_for(score: int) -> str:
    for bound, name in LEVEL_BOUNDS:
        if score >= bound:
            return name
    return "LOW"


def _short(names: list[str], limit: int = 8) -> str:
    shown = ", ".join(names[:limit])
    return shown + (f" (+{len(names) - limit} more)" if len(names) > limit else "")


def _permission_component(indicators: Mapping[str, Any], features: Mapping[str, Any]) -> Component:
    permissions = set(indicators.get("permissions", []))
    dangerous = list(indicators.get("dangerous_permission_names", []))
    evidence: list[str] = []
    points = float(min(int(features["dangerous_permissions"]), 10))
    if dangerous:
        evidence.append(f"{len(dangerous)} dangerous permission(s) requested: {_short(dangerous)}")
    if SMS_SEND in permissions and (SMS_READ in permissions or SMS_RECV in permissions):
        points += 5
        evidence.append("Requests permission to both send and read/receive SMS messages")
    if CONTACTS in permissions and (SMS_READ in permissions or CALL_LOG in permissions):
        points += 3
        evidence.append("Requests contacts together with SMS or call-log access")
    if (AUDIO in permissions or CAMERA in permissions) and INTERNET in permissions:
        points += 2
        evidence.append("Requests microphone or camera together with network access")
    return Component("dangerous_permissions", "HEURISTIC", min(points, 20.0), 20.0, evidence)


def _api_component(indicators: Mapping[str, Any], features: Mapping[str, Any]) -> Component:
    matches = list(indicators.get("suspicious_api_matches", []))
    points = 4.0 * min(int(features["suspicious_api_calls"]), 5)
    evidence = [f"References {len(matches)} watch-listed framework method(s): {_short(matches)}"] if matches else []
    return Component("suspicious_api_references", "HEURISTIC", points, 20.0, evidence)


def _structural_component(
    indicators: Mapping[str, Any], features: Mapping[str, Any], dex_available: bool
) -> Component:
    points, evidence = 0.0, []
    if int(features["activities_count"]) == 0 and (int(features["services_count"]) + int(features["receivers_count"])) > 0:
        points += 5
        evidence.append("Declares no activities but declares services or receivers (no user interface entry point)")
    if dex_available:
        matches = set(indicators.get("suspicious_api_matches", []))
        loaders = {m for m in matches if any(m.startswith(c) for c in DYNAMIC_LOADERS)}
        if loaders and REFLECTION_CALL in matches:
            points += 5
            evidence.append("References both dynamic code loading and reflection calls")
    return Component("structural_indicators", "HEURISTIC", points, 10.0, evidence)


def assess(
    extraction_status: str,
    features: Mapping[str, Optional[int]],
    indicators: Mapping[str, Any],
    malicious_probability: Optional[float],
    model_problem: Optional[str] = None,
) -> Assessment:
    """Combine observed evidence and (optionally) a model score into a result.

    `model_problem` is a short human-readable reason when no model score exists.
    """
    if extraction_status == "FAILED":
        return Assessment(
            "NEEDS REVIEW",
            ["The APK could not be analysed, so nothing can be concluded about it."],
            None, None, None, [], None,
        )

    partial = extraction_status == "PARTIAL"
    components: list[Component] = [_permission_component(indicators, features)]
    if partial:
        components.append(Component("suspicious_api_references", "HEURISTIC", None, 20.0,
                                    ["Not available: the DEX files could not be read"]))
        # Manifest-only structural indicator; DEX-based part is unavailable.
        components.append(_structural_component(indicators, features, dex_available=False))
    else:
        components.append(_api_component(indicators, features))
        components.append(_structural_component(indicators, features, dex_available=True))

    heuristic = sum(c.points for c in components if c.points is not None)
    total = heuristic
    if malicious_probability is not None and not partial:
        ml_points = round(50.0 * malicious_probability, 2)
        components.insert(0, Component(
            "ml_malware_score", "ML", ml_points, 50.0,
            [f"Model malware score {malicious_probability:.3f} (a model output, not a probability of infection)"],
        ))
        total += ml_points
        basis = "ML_AND_HEURISTIC"
    else:
        basis = "HEURISTIC_PARTIAL" if partial else "HEURISTIC_ONLY"

    score = max(0, min(100, int(round(total))))
    level = level_for(score)

    reasons: list[str] = []
    if partial:
        classification = "NEEDS REVIEW"
        reasons.append("Only part of the APK could be read, so the model was not applied.")
    elif malicious_probability is None:
        classification = "NEEDS REVIEW"
        reasons.append(model_problem or "No model score is available, so no classification was made.")
    else:
        p = malicious_probability
        low, high = REVIEW_BAND
        strong = heuristic >= STRONG_HEURISTIC_POINTS
        supported = heuristic > 0
        if low <= p <= high:
            classification = "NEEDS REVIEW"
            reasons.append(f"The model score ({p:.2f}) is in the uncertain range {low:.2f}-{high:.2f}.")
        elif p > high:
            if p >= MALICIOUS_THRESHOLD and supported:
                classification = "MALICIOUS"
                reasons.append(f"High model score ({p:.2f}) supported by observed heuristic indicators.")
            else:
                classification = "SUSPICIOUS"
                reasons.append(
                    f"Model score ({p:.2f}) is above the uncertain range"
                    + ("" if p < MALICIOUS_THRESHOLD else " but no supporting heuristic indicator was observed")
                    + "."
                )
        elif strong:
            classification = "NEEDS REVIEW"
            reasons.append(
                f"The model score ({p:.2f}) is low, but strong heuristic indicators were observed. "
                "The two sources disagree."
            )
        else:
            classification = "BENIGN"
            reasons.append(f"Low model score ({p:.2f}) and no strong heuristic indicators.")
    return Assessment(classification, reasons, score, level, basis, components, heuristic)
