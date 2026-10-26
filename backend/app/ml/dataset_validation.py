"""Dataset manifest validation.

The manifest is a CSV with one row per APK sample (see app.ml.schema.MANIFEST_COLUMNS).
Validation never repairs, guesses, or fabricates anything:

* rows with an invalid hash, label, status, or feature value are REJECTED, not fixed;
* rows without a label are never assigned one;
* failed or partial extractions are kept out of the training table, not filled in;
* duplicates are found by SHA-256, and a hash that appears with conflicting labels is
  excluded entirely because its true label is unknown.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from app.ml.schema import (
    FEATURE_NAMES,
    MANIFEST_COLUMNS,
    REQUIRED_COLUMNS,
    STATUS_FAILED,
    STATUS_OK,
    SYNTHETIC_LABEL_SOURCE_PREFIX,
    VALID_LABELS,
    VALID_STATUSES,
)
from app.services.hashing import normalize_sha256

# Below this many usable training rows, any metric is too noisy to mean much.
MIN_MEANINGFUL_TRAINING_ROWS = 200
IMBALANCE_WARNING_RATIO = 3.0

_STATUS_RANK = {"OK": 0, "PARTIAL": 1, "FAILED": 2}


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "errors": self.errors, "warnings": self.warnings, "stats": self.stats}

    def format_text(self) -> str:
        lines = ["DATASET VALIDATION: " + ("PASSED" if self.ok else "FAILED")]
        for key, value in self.stats.items():
            lines.append(f"  {key}: {value}")
        for msg in self.errors:
            lines.append(f"  ERROR:   {msg}")
        for msg in self.warnings:
            lines.append(f"  WARNING: {msg}")
        return "\n".join(lines)


@dataclass
class ValidationResult:
    report: ValidationReport
    all_rows: pd.DataFrame        # format-valid, de-duplicated rows of any extraction status
    training_rows: pd.DataFrame   # subset with status OK; feature columns typed as integers
    rejected: pd.DataFrame        # rows removed, with a reject_reason column


def parse_count(value: Any) -> Optional[int]:
    """Return a non-negative integer, or None if the value is blank or not a clean count."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value) if int(value) >= 0 else None
    if isinstance(value, (float, np.floating)):
        if math.isnan(value) or math.isinf(value) or not float(value).is_integer() or value < 0:
            return None
        return int(value)
    text = str(value).strip()
    if re.fullmatch(r"\d+", text):
        return int(text)
    if re.fullmatch(r"\d+\.0+", text):  # CSV tools often write integers as "5.0"
        return int(float(text))
    return None


def find_duplicate_hashes(hashes: "pd.Series | list[str]") -> dict[str, int]:
    """Return {sha256: occurrences} for every valid hash that appears more than once."""
    series = pd.Series(list(hashes), dtype=object).map(normalize_sha256).dropna()
    counts = series.value_counts()
    return {str(h): int(c) for h, c in counts.items() if c > 1}


def load_manifest(path: str | Path) -> pd.DataFrame:
    """Read a manifest CSV with every cell kept as text so nothing is silently coerced."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")


def validate_manifest_frame(
    df: pd.DataFrame,
    expected_extractor_version: Optional[str] = None,
) -> ValidationResult:
    report = ValidationReport()
    empty = pd.DataFrame(columns=list(MANIFEST_COLUMNS))

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        report.errors.append(f"Missing required columns: {', '.join(missing)}")
        return ValidationResult(report, empty, empty, empty)
    if len(df) == 0:
        report.errors.append("Manifest contains no rows.")
        return ValidationResult(report, empty, empty, empty)

    work = df.copy().astype(object)
    work = work.where(work.notna(), "")
    for col in ("error_code", "error_message"):
        if col not in work.columns:
            work[col] = ""
    report.stats["input_rows"] = int(len(work))

    # ---- row-level checks ------------------------------------------------
    reasons = pd.Series("", index=work.index, dtype=object)

    def mark(mask: pd.Series, reason: str) -> None:
        reasons[mask & (reasons == "")] = reason

    work["sha256"] = work["sha256"].map(lambda v: normalize_sha256(v) or "")
    mark(work["sha256"] == "", "INVALID_SHA256")

    work["label"] = work["label"].map(lambda v: str(v).strip().upper())
    mark(work["label"] == "", "UNLABELLED")
    mark(~work["label"].isin(VALID_LABELS), "INVALID_LABEL")

    for col in ("sample_id", "label_source", "extractor_version"):
        work[col] = work[col].map(lambda v: str(v).strip())
    mark(work["sample_id"] == "", "MISSING_SAMPLE_ID")
    mark(work["label_source"] == "", "MISSING_LABEL_SOURCE")

    work["extraction_status"] = work["extraction_status"].map(lambda v: str(v).strip().upper())
    mark(~work["extraction_status"].isin(VALID_STATUSES), "INVALID_STATUS")

    parsed = {name: work[name].map(parse_count) for name in FEATURE_NAMES}
    any_value = pd.Series(False, index=work.index)
    any_missing = pd.Series(False, index=work.index)
    for name in FEATURE_NAMES:
        blank = work[name].map(lambda v: str(v).strip() == "")
        any_value |= ~blank
        any_missing |= parsed[name].isna()
        # A non-blank cell that is not a clean count is always a data error.
        mark((~blank) & parsed[name].isna(), "BAD_FEATURE_VALUE")
    mark((work["extraction_status"] == STATUS_OK) & any_missing, "BAD_FEATURE_VALUE")
    mark((work["extraction_status"] == STATUS_FAILED) & any_value, "FAILED_ROW_HAS_FEATURES")

    valid = work[reasons == ""].copy()
    rejected = work[reasons != ""].copy()
    rejected["reject_reason"] = reasons[reasons != ""]

    # ---- duplicates by SHA-256 --------------------------------------------
    if len(valid):
        label_counts = valid.groupby("sha256")["label"].nunique()
        conflicting = set(label_counts[label_counts > 1].index)
        if conflicting:
            conflict_mask = valid["sha256"].isin(conflicting)
            bad = valid[conflict_mask].copy()
            bad["reject_reason"] = "CONFLICTING_LABELS"
            rejected = pd.concat([rejected, bad], ignore_index=False)
            valid = valid[~conflict_mask]
            report.warnings.append(
                f"{len(conflicting)} SHA-256 hash(es) appear with conflicting labels and were excluded."
            )

        valid["_rank"] = valid["extraction_status"].map(_STATUS_RANK)
        valid = valid.sort_values("_rank", kind="stable")
        dup_mask = valid.duplicated(subset="sha256", keep="first")
        dupes = valid[dup_mask].copy()
        dupes["reject_reason"] = "DUPLICATE_SHA256"
        rejected = pd.concat([rejected, dupes], ignore_index=False)
        valid = valid[~dup_mask].drop(columns="_rank").sort_index()
        report.stats["duplicate_rows_removed"] = int(len(dupes))
    else:
        report.stats["duplicate_rows_removed"] = 0

    rejected_counts = rejected["reject_reason"].value_counts().to_dict() if len(rejected) else {}
    report.stats["rejected_rows"] = int(len(rejected))
    if rejected_counts:
        report.stats["rejected_by_reason"] = {str(k): int(v) for k, v in rejected_counts.items()}
        non_dupe = {k: v for k, v in rejected_counts.items() if k != "DUPLICATE_SHA256"}
        if non_dupe:
            report.warnings.append(
                "Rows rejected for data problems (not repaired): "
                + ", ".join(f"{k}={v}" for k, v in sorted(non_dupe.items()))
            )

    # ---- typed outputs -----------------------------------------------------
    for name in FEATURE_NAMES:
        valid[name] = valid[name].map(parse_count).astype("Int64")
    valid = valid.reset_index(drop=True)
    training = valid[valid["extraction_status"] == STATUS_OK].copy()
    for name in FEATURE_NAMES:
        training[name] = training[name].astype("int64")
    training = training.reset_index(drop=True)

    status_counts = valid["extraction_status"].value_counts().to_dict()
    report.stats["rows_after_cleaning"] = int(len(valid))
    report.stats["status_counts"] = {str(k): int(v) for k, v in status_counts.items()}
    report.stats["training_rows"] = int(len(training))
    excluded_for_status = int(len(valid) - len(training))
    if excluded_for_status:
        report.warnings.append(
            f"{excluded_for_status} row(s) with extraction_status other than OK are kept in the "
            "manifest but excluded from training."
        )

    # ---- training-set checks -----------------------------------------------
    if len(training) == 0:
        report.errors.append("No usable training rows (extraction_status OK) remain after validation.")
        return ValidationResult(report, valid, training, rejected)

    label_counts_train = training["label"].value_counts().to_dict()
    report.stats["label_counts"] = {str(k): int(v) for k, v in label_counts_train.items()}
    if "original_category" in training.columns:
        cats = training["original_category"].replace("", "(none)").value_counts().to_dict()
        report.stats["original_category_counts"] = {str(k): int(v) for k, v in cats.items()}

    if len(label_counts_train) < 2:
        report.errors.append(
            "Training rows contain only one class "
            f"({next(iter(label_counts_train))}); a classifier needs both BENIGN and MALWARE."
        )
    else:
        majority, minority = max(label_counts_train.values()), min(label_counts_train.values())
        ratio = majority / minority
        report.stats["class_imbalance_ratio"] = round(ratio, 2)
        if ratio >= IMBALANCE_WARNING_RATIO:
            report.warnings.append(
                f"Class imbalance is {ratio:.1f}:1. Use class weights and report precision, recall, "
                "and false-positive/false-negative rates, not accuracy alone."
            )

    versions = sorted(set(training["extractor_version"]))
    report.stats["extractor_versions"] = versions
    if len(versions) > 1:
        report.errors.append(
            "Training rows were produced by different extractor versions: " + ", ".join(versions)
        )
    elif expected_extractor_version is not None and versions[0] != expected_extractor_version:
        report.errors.append(
            f"Rows were extracted with version '{versions[0]}' but the current extractor is "
            f"'{expected_extractor_version}'. Re-extract the samples."
        )

    synthetic = int(training["label_source"].str.startswith(SYNTHETIC_LABEL_SOURCE_PREFIX).sum())
    if synthetic:
        report.stats["synthetic_training_rows"] = synthetic
        report.warnings.append(
            f"{synthetic} training row(s) are marked as synthetic test data. "
            "They must never be used as evidence of detection quality."
        )

    if len(training) < MIN_MEANINGFUL_TRAINING_ROWS:
        report.warnings.append(
            f"Only {len(training)} usable training rows (fewer than {MIN_MEANINGFUL_TRAINING_ROWS}). "
            "Evaluation metrics from this little data will be too noisy to interpret."
        )

    return ValidationResult(report, valid, training, rejected)


def validate_manifest_file(
    path: str | Path, expected_extractor_version: Optional[str] = None
) -> ValidationResult:
    try:
        df = load_manifest(path)
    except FileNotFoundError:
        report = ValidationReport(errors=[f"Manifest not found: {path}"])
        empty = pd.DataFrame(columns=list(MANIFEST_COLUMNS))
        return ValidationResult(report, empty, empty, empty)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        report = ValidationReport(errors=[f"Manifest could not be read as CSV: {exc}"])
        empty = pd.DataFrame(columns=list(MANIFEST_COLUMNS))
        return ValidationResult(report, empty, empty, empty)
    return validate_manifest_frame(df, expected_extractor_version)
