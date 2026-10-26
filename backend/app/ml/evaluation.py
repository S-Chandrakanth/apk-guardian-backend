"""Evaluation of a trained model on the held-out test rows, with plain-language limits."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from app.ml.modeling import (
    ModelingError,
    category_report,
    evaluate_model,
    feature_importance,
    fit_baselines,
    malware_probability,
    review_band_metrics,
    to_xy,
)
from app.ml.predictor import Predictor
from app.ml.splitting import dataset_fingerprint

LIMITS = [
    "These numbers were measured on ONE held-out test split of THIS dataset only.",
    "They do not estimate performance on other datasets, on newer malware, or at real-world "
    "malware/benign ratios.",
    "Model probabilities are scores, not guaranteed certainty. They are not calibrated.",
    "Static analysis with eight count features cannot detect every malware sample.",
]


def build_report(
    predictor: Predictor,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int = 42,
    review_band: tuple[float, float] = (0.40, 0.60),
    synthetic: Optional[bool] = None,
) -> dict[str, Any]:
    recorded = predictor.schema.get("training_dataset", {}).get("fingerprint")
    if recorded and recorded != dataset_fingerprint(train_df):
        raise ModelingError(
            "train.csv is not the dataset this model was trained on (fingerprint mismatch). "
            "Retrain, or point to the matching processed folder."
        )
    if set(train_df["sha256"]) & set(test_df["sha256"]):
        raise ModelingError("Train and test rows share SHA-256 values; refusing to evaluate.")

    synthetic = predictor.trained_on_synthetic_test_data if synthetic is None else synthetic
    X, y = to_xy(test_df)
    proba = malware_probability(predictor.model, X)
    y_pred = (proba >= 0.5).astype(int)

    baselines = fit_baselines(train_df, seed=seed)
    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "synthetic_data": bool(synthetic),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "test_class_counts": {str(k): int(v) for k, v in test_df["label"].value_counts().items()},
        "threshold": 0.5,
        "random_forest": evaluate_model(predictor.model, test_df),
        "baselines": {name: evaluate_model(model, test_df) for name, model in baselines.items()},
        "per_original_category": category_report(test_df, y_pred),
        "needs_review_band": review_band_metrics(y, proba, *review_band),
        "permutation_importance": feature_importance(predictor.model, test_df, seed=seed),
        "limits": LIMITS,
    }
    return report


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# Model evaluation", ""]
    if report["synthetic_data"]:
        lines += [
            "> **SYNTHETIC TEST DATA. These numbers are NOT evidence of detection quality.**",
            "> They only show that the pipeline runs.",
            "",
        ]
    lines += [
        f"Train rows: {report['n_train']}  |  Test rows: {report['n_test']}  |  "
        f"Test classes: {report['test_class_counts']}",
        "",
        "## Test-set metrics (threshold 0.5, MALWARE = positive class)",
        "",
        "| Model | Accuracy | Precision | Recall | F1 | FPR | FNR | PR-AUC |",
        "|---|---|---|---|---|---|---|---|",
    ]
    rows = {"Random Forest": report["random_forest"], **{k: v for k, v in report["baselines"].items()}}
    for name, m in rows.items():
        lines.append(
            f"| {name} | {_fmt(m['accuracy'])} | {_fmt(m['precision'])} | {_fmt(m['recall'])} | "
            f"{_fmt(m['f1'])} | {_fmt(m['false_positive_rate'])} | {_fmt(m['false_negative_rate'])} | "
            f"{_fmt(m['pr_auc'])} |"
        )
    cm = report["random_forest"]["confusion_matrix"]
    lines += [
        "",
        "## Random Forest confusion matrix",
        "",
        "| | Predicted BENIGN | Predicted MALWARE |",
        "|---|---|---|",
        f"| **Actually BENIGN** | {cm['tn']} | {cm['fp']} |",
        f"| **Actually MALWARE** | {cm['fn']} | {cm['tp']} |",
        "",
        "## Per original category",
        "",
    ]
    for cat, v in report["per_original_category"].items():
        key = "recall" if "recall" in v else "false_positive_rate"
        lines.append(f"- {cat} (n={v['n']}, labelled {v['label']}): {key} = {_fmt(v[key])}")
    band = report["needs_review_band"]
    lines += [
        "",
        f"## NEEDS REVIEW band {band['band']}",
        "",
        f"{band['review_count']} of {report['n_test']} test rows fall in the band "
        f"({_fmt(band['review_rate'])}). Metrics on the remaining rows:",
        "",
        f"`{band['metrics_on_decided']}`",
        "",
        "## Permutation importance (drop in F1 when shuffled)",
        "",
    ]
    for name, v in sorted(report["permutation_importance"].items(), key=lambda kv: -kv[1]["mean_f1_drop"]):
        lines.append(f"- {name}: {v['mean_f1_drop']:.4f} (std {v['std']:.4f})")
    lines += ["", "## Limits of these results", ""] + [f"- {t}" for t in report["limits"]]
    return "\n".join(lines) + "\n"
