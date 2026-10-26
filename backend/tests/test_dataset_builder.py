"""Tests for planning and running dataset extraction (labels, duplicates, resume).

Fixture APKs are benign, synthetic, test-only files. Their label_source is marked
UNIT_TEST_SYNTHETIC so the validator flags the resulting manifest as non-evidence.
"""

import csv
import os

import pytest

from app.ml.dataset_validation import validate_manifest_file
from app.services.dataset_builder import (
    load_labels_csv,
    parse_folder_map,
    plan_samples,
    read_existing_hashes,
    run_extraction,
)
from tests.fixtures.apk_builder import build_apk

FOLDER_MAP = {"Adware": "MALWARE", "Benign": "BENIGN"}
SOURCE = "UNIT_TEST_SYNTHETIC folder names"


@pytest.fixture
def apk_tree(tmp_path):
    root = tmp_path / "apks"
    for sub in ("Adware", "Benign", "Mystery"):
        (root / sub).mkdir(parents=True)
    build_apk(root / "Adware" / "a1.apk", package="com.t.a1", permissions=["android.permission.SEND_SMS"])
    build_apk(root / "Adware" / "a2.apk", package="com.t.a2", services=[".S1", ".S2"])
    build_apk(root / "Benign" / "b1.apk", package="com.t.b1")
    # byte-identical copy of b1 -> duplicate SHA-256
    (root / "Benign" / "b1_copy.apk").write_bytes((root / "Benign" / "b1.apk").read_bytes())
    # no label mapping for this folder -> must be skipped, never guessed
    build_apk(root / "Mystery" / "m1.apk", package="com.t.m1")
    # byte-identical file placed under two folders with different labels -> conflict
    build_apk(root / "Adware" / "c.apk", package="com.t.c")
    (root / "Benign" / "c.apk").write_bytes((root / "Adware" / "c.apk").read_bytes())
    # corrupted file with a label
    (root / "Benign" / "broken.apk").write_bytes(b"PK\x03\x04" + os.urandom(2000))
    return root


def test_plan_labels_only_from_explicit_map(apk_tree):
    plan = plan_samples(apk_tree, folder_map=FOLDER_MAP, label_source=SOURCE)
    ids = {s.sample_id: s.info.label for s in plan.to_extract}
    assert ids["Adware/a1"] == "MALWARE" and ids["Benign/b1"] == "BENIGN"
    assert "Mystery/m1" not in ids
    assert plan.unlabelled == ["Mystery/m1.apk"]
    assert all(s.info.label_source == SOURCE for s in plan.to_extract)


def test_plan_detects_duplicates_and_conflicts(apk_tree):
    plan = plan_samples(apk_tree, folder_map=FOLDER_MAP, label_source=SOURCE)
    assert len(plan.duplicates) == 1
    assert sorted(plan.duplicates[0][1]) == ["Benign/b1.apk", "Benign/b1_copy.apk"]
    assert len(plan.conflicts) == 1
    assert plan.conflicts[0][1] == ["BENIGN", "MALWARE"]
    extracted_ids = {s.sample_id for s in plan.to_extract}
    assert not any(i.endswith("/c") for i in extracted_ids)  # conflicting hash excluded
    assert len({s.sha256 for s in plan.to_extract}) == len(plan.to_extract)  # one row per hash


def test_plan_requires_a_label_method_and_a_source(apk_tree):
    with pytest.raises(ValueError, match="never guessed"):
        plan_samples(apk_tree)
    with pytest.raises(ValueError, match="label-source"):
        plan_samples(apk_tree, folder_map=FOLDER_MAP)


def test_plan_limit(apk_tree):
    assert len(plan_samples(apk_tree, folder_map=FOLDER_MAP, label_source=SOURCE, limit=2).to_extract) == 2


def test_parse_folder_map():
    assert parse_folder_map("Adware=malware, Benign=BENIGN") == FOLDER_MAP
    for bad in ("", "Adware", "Adware=TROJAN", "=MALWARE"):
        with pytest.raises(ValueError):
            parse_folder_map(bad)


def test_labels_csv_loading_and_errors(tmp_path):
    good = tmp_path / "labels.csv"
    good.write_text("filename,label,original_category,label_source\nx.apk,malware,Banking,paper table 2\n")
    info = load_labels_csv(good, None)["x.apk"]
    assert (info.label, info.original_category, info.label_source) == ("MALWARE", "Banking", "paper table 2")

    bad_label = tmp_path / "bad.csv"
    bad_label.write_text("filename,label\nx.apk,maybe\n")
    with pytest.raises(ValueError, match="not in"):
        load_labels_csv(bad_label, "src")

    no_source = tmp_path / "nosrc.csv"
    no_source.write_text("filename,label\nx.apk,BENIGN\n")
    with pytest.raises(ValueError, match="label_source"):
        load_labels_csv(no_source, None)
    assert load_labels_csv(no_source, "given on cli")["x.apk"].label_source == "given on cli"

    wrong_cols = tmp_path / "cols.csv"
    wrong_cols.write_text("name,class\nx.apk,BENIGN\n")
    with pytest.raises(ValueError, match="filename,label"):
        load_labels_csv(wrong_cols, "src")


def test_labels_csv_can_label_by_basename(apk_tree, tmp_path):
    labels = tmp_path / "labels.csv"
    labels.write_text("filename,label\nm1.apk,BENIGN\n")
    plan = plan_samples(apk_tree, labels_csv=load_labels_csv(labels, "csv source"))
    assert [s.sample_id for s in plan.to_extract] == ["Mystery/m1"]
    assert plan.to_extract[0].info.label == "BENIGN"


def test_end_to_end_extraction_manifest_validation_and_resume(apk_tree, tmp_path):
    manifest = tmp_path / "out" / "manifest.csv"
    plan = plan_samples(apk_tree, folder_map=FOLDER_MAP, label_source=SOURCE)
    summary = run_extraction(plan.to_extract, manifest, workers=2, timeout_seconds=120)

    assert summary["status_OK"] == 3  # a1, a2, b1
    assert summary["status_FAILED"] == 1  # broken.apk
    assert summary["error_CORRUPT_ZIP"] == 1

    rows = {r["sample_id"]: r for r in csv.DictReader(open(manifest, encoding="utf-8"))}
    assert set(rows) == {"Adware/a1", "Adware/a2", "Benign/b1", "Benign/broken"}
    assert rows["Adware/a1"]["permissions_count"] == "1"
    assert rows["Adware/a1"]["dangerous_permissions"] == "1"  # SEND_SMS
    assert rows["Adware/a2"]["services_count"] == "2"
    broken = rows["Benign/broken"]
    assert broken["extraction_status"] == "FAILED"
    assert broken["error_code"] == "CORRUPT_ZIP"
    assert all(broken[f] == "" for f in ("permissions_count", "api_calls_count", "strings_count"))  # blank, not 0

    result = validate_manifest_file(manifest)
    assert len(result.training_rows) == 3
    assert (result.all_rows["extraction_status"] == "FAILED").sum() == 1
    assert any("synthetic" in w for w in result.report.warnings)

    # resume: everything already in the manifest is skipped
    again = plan_samples(apk_tree, folder_map=FOLDER_MAP, label_source=SOURCE,
                         existing_hashes=read_existing_hashes(manifest))
    assert again.to_extract == []
    assert again.skipped_existing == 4


def test_run_extraction_summary_always_has_every_status_key(apk_tree, tmp_path):
    """Regression: the summary used to omit statuses that did not occur (KeyError: 'status_OK')."""
    keys = {"status_OK", "status_PARTIAL", "status_FAILED"}

    empty = run_extraction([], tmp_path / "none.csv")
    assert keys <= set(empty) and all(empty[k] == 0 for k in keys)

    plan = plan_samples(apk_tree, folder_map=FOLDER_MAP, label_source=SOURCE)
    only_broken = [s for s in plan.to_extract if s.sample_id == "Benign/broken"]
    assert len(only_broken) == 1
    summary = run_extraction(only_broken, tmp_path / "broken_only.csv", workers=1, timeout_seconds=120)
    assert summary["status_OK"] == 0
    assert summary["status_PARTIAL"] == 0
    assert summary["status_FAILED"] == 1
    assert summary["error_CORRUPT_ZIP"] == 1


def test_run_extraction_refuses_to_write_rows_when_androguard_is_unusable(apk_tree, tmp_path, monkeypatch):
    from app.services import dataset_builder as db
    from app.services.apk_features import ExtractorEnvironmentError

    def broken():
        raise ExtractorEnvironmentError("Androguard is not installed")

    monkeypatch.setattr(db, "load_androguard", broken)
    plan = plan_samples(apk_tree, folder_map=FOLDER_MAP, label_source=SOURCE)
    manifest = tmp_path / "never.csv"
    with pytest.raises(ExtractorEnvironmentError):
        run_extraction(plan.to_extract, manifest)
    assert not manifest.exists()
