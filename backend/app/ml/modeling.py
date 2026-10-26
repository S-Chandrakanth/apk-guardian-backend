"""Random Forest baseline: tuning, fitting, metrics, and baselines.

This module only classifies. It contains nothing about explanations, risk levels, or the
API. The AI explanation component is a separate module that consumes the classifier's
output and can never change a prediction.

Preprocessing note: Random Forests are insensitive to feature scaling and the validated
inputs are non-negative integers with no missing values, so the forest consumes the raw
counts directly. The logistic-regression baseline log-scales and standardises because
linear models need it.
"""

from __future__ import annotations

import platform
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from app.ml.schema import FEATURE_NAMES, LABEL_BENIGN, LABEL_MALWARE
from app.ml.splitting import vector_groups

DEFAULT_GRID: dict[str, list[Any]] = {
    "rf__n_estimators": [200, 400],
    "rf__max_depth": [None, 12],
    "rf__min_samples_leaf": [1, 3],
}
FIXED_PARAMS: dict[str, Any] = {"rf__n_estimators": 300, "rf__max_depth": None, "rf__min_samples_leaf": 1}


class ModelingError(ValueError):
    """Training or evaluation could not be completed safely."""


# --------------------------------------------------------------------------
# Data helpers
# --------------------------------------------------------------------------
def to_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Features in the canonical column order, and y with MALWARE = 1, BENIGN = 0."""
    X = df[list(FEATURE_NAMES)].astype("int64")
    y = (df["label"] == LABEL_MALWARE).astype(int).to_numpy()
    return X, y


def malware_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    classes = list(model.classes_)
    if 1 not in classes:
        raise ModelingError("Model was not trained with a MALWARE class.")
    return model.predict_proba(X)[:, classes.index(1)]


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def build_rf_pipeline(seed: int = 42, **rf_params: Any) -> Pipeline:
    forest = RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=-1, **rf_params)
    return Pipeline([("rf", forest)])


def tune_and_fit(
    train_df: pd.DataFrame,
    seed: int = 42,
    cv_folds: int = 5,
    grid: Optional[Mapping[str, Sequence[Any]]] = None,
    tune: bool = True,
) -> tuple[Pipeline, dict[str, Any]]:
    """Fit the Random Forest on TRAINING rows only.

    Hyper-parameters are chosen by cross-validation inside the training rows. The CV folds
    keep identical feature vectors together (same reasoning as the train/test split), so the
    CV score is not inflated by near-duplicates. The held-out test rows are never touched here.
    """
    X, y = to_xy(train_df)
    if len(set(y)) < 2:
        raise ModelingError("Training rows contain only one class.")
    info: dict[str, Any] = {
        "n_train": int(len(train_df)),
        "train_class_counts": {
            LABEL_BENIGN: int((y == 0).sum()),
            LABEL_MALWARE: int((y == 1).sum()),
        },
        "seed": seed,
        "class_weight": "balanced",
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
    }

    if not tune:
        params = dict(FIXED_PARAMS)
        pipe = build_rf_pipeline(seed).set_params(**params)
        pipe.fit(X, y)
        info.update(tuned=False, best_params=_plain(params))
        return pipe, info

    min_class = int(min((y == 0).sum(), (y == 1).sum()))
    if min_class < cv_folds:
        raise ModelingError(
            f"The smaller class has {min_class} training samples, fewer than cv_folds={cv_folds}."
        )
    groups = vector_groups(train_df)
    cv = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    search = GridSearchCV(
        build_rf_pipeline(seed),
        dict(grid or DEFAULT_GRID),
        scoring="f1_macro",
        cv=cv,
        refit=True,
        n_jobs=1,
    )
    try:
        search.fit(X, y, groups=groups)
    except ValueError as exc:
        raise ModelingError(f"Cross-validation failed: {exc}") from exc

    results = pd.DataFrame(search.cv_results_)
    info.update(
        tuned=True,
        cv_folds=cv_folds,
        cv_scoring="f1_macro (cross-validation on TRAINING rows only; this is not a test result)",
        best_params=_plain(search.best_params_),
        best_cv_f1_macro_mean=float(search.best_score_),
        best_cv_f1_macro_std=float(results.loc[search.best_index_, "std_test_score"]),
        cv_candidates=[
            {
                "params": _plain(row["params"]),
                "mean_f1_macro": float(row["mean_test_score"]),
                "std_f1_macro": float(row["std_test_score"]),
            }
            for _, row in results.iterrows()
        ],
    )
    return search.best_estimator_, info


def _plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def _ratio(num: float, den: float) -> Optional[float]:
    return float(num / den) if den else None


def binary_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], y_score: Optional[Sequence[float]] = None
) -> dict[str, Any]:
    """Metrics with MALWARE as the positive class. Undefined values are None, never 0."""
    y_true, y_pred = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel())
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    f1 = None if precision is None or recall is None or (precision + recall) == 0 else (
        2 * precision * recall / (precision + recall)
    )
    out: dict[str, Any] = {
        "n": int(len(y_true)),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "accuracy": _ratio(tp + tn, len(y_true)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": _ratio(fp, fp + tn),
        "false_negative_rate": _ratio(fn, fn + tp),
        "roc_auc": None,
        "pr_auc": None,
    }
    if y_score is not None and len(set(y_true.tolist())) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
    return out


def category_report(test_df: pd.DataFrame, y_pred: Sequence[int]) -> dict[str, Any]:
    """Per original category: how often the model said MALWARE.

    For categories labelled MALWARE this is the recall for that category. For BENIGN
    categories it is the false-positive rate. One aggregate number can hide a weak category.
    """
    out: dict[str, Any] = {}
    frame = test_df.assign(_pred=np.asarray(y_pred, dtype=int))
    frame["original_category"] = frame["original_category"].replace("", "(none)")
    for category, part in frame.groupby("original_category"):
        label = part["label"].mode().iloc[0]
        rate = float(part["_pred"].mean())
        out[str(category)] = {
            "n": int(len(part)),
            "label": label,
            ("recall" if label == LABEL_MALWARE else "false_positive_rate"): rate,
        }
    return out


def review_band_metrics(
    y_true: Sequence[int], proba: Sequence[float], low: float, high: float
) -> dict[str, Any]:
    """Draft NEEDS-REVIEW behaviour: scores in [low, high] are not decided automatically."""
    y_true, proba = np.asarray(y_true, dtype=int), np.asarray(proba, dtype=float)
    review = (proba >= low) & (proba <= high)
    decided = ~review
    out: dict[str, Any] = {
        "band": [low, high],
        "review_count": int(review.sum()),
        "review_rate": float(review.mean()) if len(proba) else None,
        "decided_count": int(decided.sum()),
        "metrics_on_decided": None,
    }
    if decided.any():
        pred = (proba[decided] > high).astype(int)
        out["metrics_on_decided"] = binary_metrics(y_true[decided], pred, proba[decided])
    return out


# --------------------------------------------------------------------------
# Baselines and importance
# --------------------------------------------------------------------------
def fit_baselines(train_df: pd.DataFrame, seed: int = 42) -> dict[str, Pipeline]:
    X, y = to_xy(train_df)
    majority = DummyClassifier(strategy="most_frequent").fit(X, y)
    logistic = Pipeline(
        [
            ("log", FunctionTransformer(np.log1p, validate=False)),
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)),
        ]
    ).fit(X, y)
    return {"majority_class": majority, "logistic_regression": logistic}


def evaluate_model(model: Any, df: pd.DataFrame, threshold: float = 0.5) -> dict[str, Any]:
    X, y = to_xy(df)
    proba = malware_probability(model, X)
    return binary_metrics(y, (proba >= threshold).astype(int), proba)


def feature_importance(model: Pipeline, test_df: pd.DataFrame, seed: int = 42) -> dict[str, Any]:
    """Permutation importance on the held-out test rows (drop in F1 when a feature is shuffled)."""
    X, y = to_xy(test_df)
    result = permutation_importance(model, X, y, scoring="f1", n_repeats=10, random_state=seed, n_jobs=1)
    return {
        name: {"mean_f1_drop": float(m), "std": float(s)}
        for name, m, s in zip(FEATURE_NAMES, result.importances_mean, result.importances_std)
    }
