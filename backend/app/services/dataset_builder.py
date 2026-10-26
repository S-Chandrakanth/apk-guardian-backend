"""Build a dataset manifest from a folder of APK files you are licensed to hold.

Rules:
* Nothing is ever labelled automatically. A file gets a label only from an explicit
  labels CSV or an explicit folder-to-label map supplied by the user. Files without a
  label are skipped and reported.
* Duplicate files (same SHA-256) are extracted once. A hash that appears under
  conflicting labels is excluded and reported.
* APKs are only parsed, never executed.
"""

from __future__ import annotations

import csv
import queue
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from app.ml.schema import MANIFEST_COLUMNS, VALID_LABELS, VALID_STATUSES
from app.services.apk_features import (
    ExtractionLimits,
    ExtractionResult,
    IsolatedExtractor,
    load_androguard,
)
from app.services.hashing import sha256_file


@dataclass
class LabelInfo:
    label: str
    original_category: str
    label_source: str


@dataclass
class Sample:
    sample_id: str
    path: Path
    sha256: str
    info: LabelInfo


@dataclass
class SamplePlan:
    to_extract: list[Sample] = field(default_factory=list)
    duplicates: list[tuple[str, list[str]]] = field(default_factory=list)
    conflicts: list[tuple[str, list[str], list[str]]] = field(default_factory=list)
    unlabelled: list[str] = field(default_factory=list)
    skipped_existing: int = 0
    total_files: int = 0


def collect_apks(apk_dir: Path) -> list[Path]:
    return sorted(p for p in Path(apk_dir).rglob("*") if p.is_file() and p.suffix.lower() == ".apk")


def parse_folder_map(text: str) -> dict[str, str]:
    """Parse 'Adware=MALWARE,Benign=BENIGN' into {'Adware': 'MALWARE', 'Benign': 'BENIGN'}."""
    mapping: dict[str, str] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Bad folder-map entry '{part}'. Use Folder=LABEL.")
        folder, label = (x.strip() for x in part.split("=", 1))
        label = label.upper()
        if not folder or label not in VALID_LABELS:
            raise ValueError(f"Bad folder-map entry '{part}'. Label must be one of {VALID_LABELS}.")
        mapping[folder] = label
    if not mapping:
        raise ValueError("folder-map is empty.")
    return mapping


def load_labels_csv(path: Path, default_source: Optional[str]) -> dict[str, LabelInfo]:
    """Read a labels CSV with columns: filename, label, [original_category], [label_source].

    'filename' may be the path relative to the APK folder (with / separators) or just the
    base file name.
    """
    out: dict[str, LabelInfo] = {}
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        needed = {"filename", "label"}
        if reader.fieldnames is None or not needed.issubset(reader.fieldnames):
            raise ValueError("labels CSV needs at least the columns: filename,label")
        for line_no, row in enumerate(reader, start=2):
            name = (row.get("filename") or "").strip().replace("\\", "/")
            label = (row.get("label") or "").strip().upper()
            if not name:
                raise ValueError(f"labels CSV line {line_no}: empty filename")
            if label not in VALID_LABELS:
                raise ValueError(f"labels CSV line {line_no}: label '{label}' is not in {VALID_LABELS}")
            source = (row.get("label_source") or "").strip() or (default_source or "")
            if not source:
                raise ValueError(
                    f"labels CSV line {line_no}: no label_source column value and no --label-source given"
                )
            out[name] = LabelInfo(label, (row.get("original_category") or "").strip(), source)
    return out


def _resolve(
    rel: str,
    labels_csv: Optional[dict[str, LabelInfo]],
    folder_map: Optional[dict[str, str]],
    label_source: Optional[str],
) -> Optional[LabelInfo]:
    if labels_csv is not None:
        info = labels_csv.get(rel) or labels_csv.get(rel.rsplit("/", 1)[-1])
        if info is not None:
            return info
    if folder_map is not None and "/" in rel:
        folder = rel.split("/", 1)[0]
        label = folder_map.get(folder)
        if label is None:
            lowered = {k.lower(): v for k, v in folder_map.items()}
            label = lowered.get(folder.lower())
        if label is not None:
            return LabelInfo(label, folder, label_source or "")
    return None


def plan_samples(
    apk_dir: Path,
    *,
    labels_csv: Optional[dict[str, LabelInfo]] = None,
    folder_map: Optional[dict[str, str]] = None,
    label_source: Optional[str] = None,
    existing_hashes: frozenset[str] = frozenset(),
    limit: Optional[int] = None,
) -> SamplePlan:
    if labels_csv is None and folder_map is None:
        raise ValueError("Provide a labels CSV or a folder map. Unlabelled APKs are never guessed.")
    if folder_map is not None and not (label_source or "").strip():
        raise ValueError("--label-source is required with a folder map so every label has a provenance.")

    apk_dir = Path(apk_dir)
    plan = SamplePlan()
    groups: dict[str, list[tuple[str, Path, LabelInfo]]] = defaultdict(list)
    files = collect_apks(apk_dir)
    plan.total_files = len(files)
    for path in files:
        rel = path.relative_to(apk_dir).as_posix()
        info = _resolve(rel, labels_csv, folder_map, label_source)
        if info is None:
            plan.unlabelled.append(rel)
            continue
        groups[sha256_file(path)].append((rel, path, info))

    for sha, members in sorted(groups.items()):
        labels = sorted({m[2].label for m in members})
        if len(labels) > 1:
            plan.conflicts.append((sha, labels, [m[0] for m in members]))
            continue
        if len(members) > 1:
            plan.duplicates.append((sha, [m[0] for m in members]))
        if sha in existing_hashes:
            plan.skipped_existing += 1
            continue
        rel, path, info = members[0]
        sample_id = rel[:-4] if rel.lower().endswith(".apk") else rel
        plan.to_extract.append(Sample(sample_id, path, sha, info))

    if limit is not None:
        plan.to_extract = plan.to_extract[:limit]
    return plan


def read_existing_hashes(manifest: Path) -> frozenset[str]:
    if not Path(manifest).exists():
        return frozenset()
    with open(manifest, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "sha256" not in reader.fieldnames:
            return frozenset()
        return frozenset((row.get("sha256") or "").strip().lower() for row in reader)


def _clean_message(message: Optional[str]) -> str:
    return " ".join((message or "").split())[:300]


def sample_to_row(sample: Sample, result: ExtractionResult) -> dict[str, object]:
    row: dict[str, object] = {
        "sample_id": sample.sample_id,
        "sha256": sample.sha256,
        "label": sample.info.label,
        "original_category": sample.info.original_category,
        "label_source": sample.info.label_source,
        "extractor_version": result.extractor_version,
        "extraction_status": result.status,
        "error_code": result.error_code or "",
        "error_message": _clean_message(result.error_message),
    }
    for name, value in result.features.items():
        row[name] = "" if value is None else value  # blank, never 0, when unknown
    return row


def append_rows(manifest: Path, rows: list[dict[str, object]]) -> None:
    manifest = Path(manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    new_file = not manifest.exists() or manifest.stat().st_size == 0
    with open(manifest, "a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(MANIFEST_COLUMNS))
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def run_extraction(
    samples: list[Sample],
    manifest: Path,
    *,
    workers: int = 2,
    timeout_seconds: float = 120.0,
    limits: Optional[ExtractionLimits] = None,
    progress: Optional[Callable[[int, int, Sample, ExtractionResult], None]] = None,
) -> dict[str, int]:
    """Extract every sample (in isolated worker processes) and append rows to the manifest.

    Returns a summary with a fixed set of keys: status_OK, status_PARTIAL and status_FAILED
    are ALWAYS present (0 when none), plus error_<CODE> keys for the error codes that occurred.
    Raises ExtractorEnvironmentError before touching the manifest if Androguard is unusable.
    """
    load_androguard()  # fail fast on a broken installation instead of writing bogus FAILED rows
    summary: dict[str, int] = defaultdict(int)
    for status in VALID_STATUSES:
        summary[f"status_{status}"] = 0
    if not samples:
        return dict(summary)

    workers = max(1, min(workers, len(samples)))
    pool_q: "queue.Queue[IsolatedExtractor]" = queue.Queue()
    extractors = [IsolatedExtractor(limits=limits, timeout_seconds=timeout_seconds) for _ in range(workers)]
    for ex in extractors:
        pool_q.put(ex)
    lock = threading.Lock()
    done = [0]

    def job(sample: Sample) -> None:
        ex = pool_q.get()
        try:
            result = ex.extract(sample.path)
        finally:
            pool_q.put(ex)
        with lock:
            append_rows(manifest, [sample_to_row(sample, result)])
            summary[f"status_{result.status}"] += 1
            if result.error_code:
                summary[f"error_{result.error_code}"] += 1
            done[0] += 1
            if progress:
                progress(done[0], len(samples), sample, result)

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(job, samples))
    finally:
        for ex in extractors:
            ex.close()
    return dict(summary)
