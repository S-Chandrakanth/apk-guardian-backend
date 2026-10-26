"""Leakage-aware train/test splitting.

Samples whose eight feature values are IDENTICAL are treated as one group and are always
placed on the same side of the split. Identical vectors are the closest thing to "the same
or a very closely related APK" that these coarse features can reveal, and letting them sit
in both train and test would make the test score look better than it is.

The split is also stratified by label so both classes appear in both parts.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from app.ml.dataset_validation import parse_count
from app.ml.schema import FEATURE_NAMES, IDENTITY_COLUMNS, LABEL_BENIGN, LABEL_MALWARE

PROCESSED_COLUMNS: tuple[str, ...] = (*IDENTITY_COLUMNS, *FEATURE_NAMES)


class SplitError(ValueError):
    """The data cannot be split safely."""


@dataclass
class SplitResult:
    train: pd.DataFrame
    test: pd.DataFrame
    report: dict[str, Any]


def vector_groups(df: pd.DataFrame) -> np.ndarray:
    """Integer group id per row; rows with identical feature vectors share an id."""
    columns = [df[name].astype("int64").tolist() for name in FEATURE_NAMES]
    codes: dict[tuple, int] = {}
    return np.array([codes.setdefault(key, len(codes)) for key in zip(*columns)], dtype=int)


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """Order-independent SHA-256 over (sha256, label) pairs; identifies exactly which samples."""
    lines = sorted(f"{h}:{l}" for h, l in zip(df["sha256"], df["label"]))
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def prepare_split(training_rows: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> SplitResult:
    if not 0.05 <= test_size <= 0.5:
        raise SplitError("test_size must be between 0.05 and 0.5")
    if training_rows["sha256"].duplicated().any():
        raise SplitError("Duplicate SHA-256 values reached the splitter; validate and de-duplicate first.")
    labels = training_rows["label"]
    counts = labels.value_counts()
    if set(counts.index) != {LABEL_BENIGN, LABEL_MALWARE}:
        raise SplitError("Both BENIGN and MALWARE rows are required to make a split.")

    n_splits = max(2, int(round(1.0 / test_size)))
    if counts.min() < n_splits:
        raise SplitError(
            f"The smaller class has only {int(counts.min())} samples; at least {n_splits} are needed "
            f"for a {test_size:.0%} test split."
        )

    df = training_rows.reset_index(drop=True)
    groups = vector_groups(df)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, test_idx = next(iter(splitter.split(df, df["label"], groups)))
    train, test = df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)

    for name, part in (("train", train), ("test", test)):
        if set(part["label"]) != {LABEL_BENIGN, LABEL_MALWARE}:
            raise SplitError(
                f"The {name} part lacks one of the classes. Too few samples, or too many identical "
                "feature vectors, to split safely."
            )

    shared_sha = set(train["sha256"]) & set(test["sha256"])
    shared_vec = set(vector_groups(pd.concat([train, test], ignore_index=True))[: len(train)]) & set(
        vector_groups(pd.concat([train, test], ignore_index=True))[len(train):]
    )
    if shared_sha or shared_vec:  # cannot happen by construction; guard anyway
        raise SplitError("Leakage detected between train and test parts.")

    group_sizes = pd.Series(groups).value_counts()
    warnings: list[str] = []
    if group_sizes.iloc[0] / len(df) > 0.10:
        warnings.append(
            f"{group_sizes.iloc[0]} samples ({group_sizes.iloc[0] / len(df):.0%}) share one identical "
            "feature vector. The eight features are too coarse to tell them apart."
        )

    def counts_of(part: pd.DataFrame) -> dict[str, int]:
        return {str(k): int(v) for k, v in part["label"].value_counts().items()}

    report = {
        "seed": seed,
        "requested_test_size": test_size,
        "method": f"StratifiedGroupKFold(n_splits={n_splits}, shuffle=True), first fold held out",
        "grouping": "rows with identical 8-feature vectors are kept on the same side of the split",
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "actual_test_fraction": round(len(test) / len(df), 4),
        "train_class_counts": counts_of(train),
        "test_class_counts": counts_of(test),
        "n_feature_vector_groups": int(len(group_sizes)),
        "largest_group_size": int(group_sizes.iloc[0]),
        "leakage_check": {"shared_sha256": 0, "shared_feature_vectors": 0},
        "train_fingerprint": dataset_fingerprint(train),
        "test_fingerprint": dataset_fingerprint(test),
        "warnings": warnings,
    }
    return SplitResult(train, test, report)


def write_processed(df: pd.DataFrame, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df[list(PROCESSED_COLUMNS)].to_csv(path, index=False, encoding="utf-8")


def load_processed(path: str | Path) -> pd.DataFrame:
    """Read train.csv/test.csv back with strictly validated integer feature columns."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    missing = [c for c in PROCESSED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    for name in FEATURE_NAMES:
        parsed = df[name].map(parse_count)
        if parsed.isna().any():
            raise ValueError(f"{path}: column {name} contains blank or invalid values")
        df[name] = parsed.astype("int64")
    if not df["label"].isin([LABEL_BENIGN, LABEL_MALWARE]).all():
        raise ValueError(f"{path}: unexpected label values")
    return df
