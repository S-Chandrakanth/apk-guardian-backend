"""Phase 3 tests: splitting, training, saving/loading, prediction, metrics, evaluation.

The data in these tests is SYNTHETIC and exists only to exercise the code paths. The tests
check mechanics (no leakage, schema checks, correct metric arithmetic). They say nothing
about real detection quality.
"""

import json

import numpy as np
import pandas as pd
import pytest

from app.ml import feature_lists
from app.ml.evaluation import build_report, render_markdown
from app.ml.modeling import ModelingError, binary_metrics, review_band_metrics, tune_and_fit
from app.ml.predictor import (
    IncompleteFeaturesError,
    InvalidFeaturesError,
    ModelSchemaError,
    Predictor,
    save_bundle,
)
from app.ml.schema import FEATURE_NAMES, IDENTITY_COLUMNS
from app.ml.splitting import (
    SplitError,
    dataset_fingerprint,
    load_processed,
    prepare_split,
    vector_groups,
    write_processed,
)
from tests.conftest import sha


def synthetic_frame(n_per_class=60, seed=0, duplicate_vectors=0):
    rng = np.random.default_rng(seed)
    rows = []
    for label, base in (("BENIGN", 8), ("MALWARE", 14)):
        for i in range(n_per_class):
            vals = rng.poisson(base, len(FEATURE_NAMES)) + rng.integers(0, 40, len(FEATURE_NAMES))
            rows.append({
                "sample_id": f"{label}-{i}", "sha256": sha(f"{label}-{i}-{seed}"), "label": label,
                "original_category": "Adware" if label == "MALWARE" else "Benign",
                "label_source": "UNIT_TEST_SYNTHETIC", "extractor_version": feature_lists.extractor_version(),
                "extraction_status": "OK", **{n: int(v) for n, v in zip(FEATURE_NAMES, vals)},
            })
    df = pd.DataFrame(rows)
    if duplicate_vectors:
        extra = df.iloc[:duplicate_vectors].copy()
        extra["sha256"] = [sha(f"dup-{i}") for i in range(duplicate_vectors)]
        extra["sample_id"] = [f"dup-{i}" for i in range(duplicate_vectors)]
        df = pd.concat([df, extra], ignore_index=True)
    return df


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    df = synthetic_frame(duplicate_vectors=10)
    split = prepare_split(df, 0.2, 42)
    model, info = tune_and_fit(split.train, seed=42, cv_folds=3,
                               grid={"rf__n_estimators": [20], "rf__max_depth": [None, 5]})
    model_dir = tmp_path_factory.mktemp("model")
    save_bundle(model, model_dir, info, {"fingerprint": dataset_fingerprint(split.train), "synthetic": True})
    return split, model, info, model_dir


# ---------------- splitting ----------------
def test_split_has_no_hash_or_vector_leakage():
    df = synthetic_frame(duplicate_vectors=12)
    split = prepare_split(df, 0.2, 1)
    assert not set(split.train["sha256"]) & set(split.test["sha256"])
    train_vecs = {tuple(r) for r in split.train[list(FEATURE_NAMES)].to_numpy()}
    test_vecs = {tuple(r) for r in split.test[list(FEATURE_NAMES)].to_numpy()}
    assert not train_vecs & test_vecs
    assert len(split.train) + len(split.test) == len(df)


def test_split_is_stratified_and_reproducible():
    df = synthetic_frame()
    a, b = prepare_split(df, 0.2, 7), prepare_split(df, 0.2, 7)
    assert a.report["train_fingerprint"] == b.report["train_fingerprint"]
    for part in (a.train, a.test):
        assert set(part["label"]) == {"BENIGN", "MALWARE"}
    assert abs(a.report["actual_test_fraction"] - 0.2) < 0.08
    assert prepare_split(df, 0.2, 8).report["train_fingerprint"] != a.report["train_fingerprint"]


def test_split_refuses_unsafe_input():
    df = synthetic_frame(n_per_class=3)
    with pytest.raises(SplitError):
        prepare_split(df, 0.2)
    with pytest.raises(SplitError):
        prepare_split(synthetic_frame()[lambda d: d.label == "MALWARE"])
    dup = pd.concat([synthetic_frame(), synthetic_frame().iloc[:1]])
    with pytest.raises(SplitError, match="Duplicate"):
        prepare_split(dup)
    with pytest.raises(SplitError):
        prepare_split(synthetic_frame(), test_size=0.9)


def test_vector_groups_group_identical_rows():
    df = synthetic_frame(n_per_class=5, duplicate_vectors=2)
    groups = vector_groups(df)
    assert groups[0] == groups[len(df) - 2] and groups[1] == groups[len(df) - 1]


def test_processed_roundtrip_and_strict_loading(tmp_path):
    df = synthetic_frame(n_per_class=5)
    write_processed(df, tmp_path / "t.csv")
    back = load_processed(tmp_path / "t.csv")
    assert list(back.columns[: len(IDENTITY_COLUMNS)]) == list(IDENTITY_COLUMNS)
    assert (back[list(FEATURE_NAMES)].to_numpy() == df[list(FEATURE_NAMES)].to_numpy()).all()
    bad = pd.read_csv(tmp_path / "t.csv", dtype=str)
    bad.loc[0, "strings_count"] = ""
    bad.to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError, match="blank or invalid"):
        load_processed(tmp_path / "bad.csv")


# ---------------- training ----------------
def test_training_records_cv_on_train_only_and_class_weight(trained):
    split, _, info, _ = trained
    assert info["tuned"] and info["class_weight"] == "balanced"
    assert info["n_train"] == len(split.train)
    assert "not a test result" in info["cv_scoring"]


def test_training_rejects_single_class_and_tiny_classes():
    df = synthetic_frame()
    with pytest.raises(ModelingError):
        tune_and_fit(df[df.label == "MALWARE"], tune=False)
    tiny = pd.concat([df[df.label == "BENIGN"].iloc[:2], df[df.label == "MALWARE"]])
    with pytest.raises(ModelingError, match="fewer than cv_folds"):
        tune_and_fit(tiny, cv_folds=5)


def test_training_is_reproducible():
    split = prepare_split(synthetic_frame(), 0.2, 3)
    grid = {"rf__n_estimators": [15], "rf__max_depth": [None]}
    m1, _ = tune_and_fit(split.train, seed=5, cv_folds=3, grid=grid)
    m2, _ = tune_and_fit(split.train, seed=5, cv_folds=3, grid=grid)
    X = split.test[list(FEATURE_NAMES)]
    assert np.array_equal(m1.predict_proba(X), m2.predict_proba(X))


# ---------------- save / load / predict ----------------
def test_saved_model_reloads_and_matches_in_memory_model(trained):
    split, model, _, model_dir = trained
    predictor = Predictor.load(model_dir)
    X = split.test[list(FEATURE_NAMES)]
    assert np.allclose(model.predict_proba(X)[:, 1], predictor.model.predict_proba(X)[:, 1])
    schema = json.loads((model_dir / "feature_schema.json").read_text())
    assert schema["feature_names"] == list(FEATURE_NAMES)
    assert schema["extractor_version"] == feature_lists.extractor_version()
    assert predictor.trained_on_synthetic_test_data is True


def test_prediction_output(trained):
    split, _, _, model_dir = trained
    predictor = Predictor.load(model_dir)
    features = {n: int(split.test.iloc[0][n]) for n in FEATURE_NAMES}
    p = predictor.predict(features)
    assert 0.0 <= p.malware_probability <= 1.0
    assert p.benign_probability == pytest.approx(1 - p.malware_probability)
    assert p.raw_label in ("BENIGN", "MALWARE")
    assert p.trained_on_synthetic_test_data is True
    assert predictor.predict({**features, "permissions_count": "7"}).malware_probability >= 0  # numeric text ok


def test_prediction_refuses_missing_or_invalid_features(trained):
    _, _, _, model_dir = trained
    predictor = Predictor.load(model_dir)
    ok = {n: 5 for n in FEATURE_NAMES}
    with pytest.raises(IncompleteFeaturesError):
        predictor.predict({k: v for k, v in ok.items() if k != "strings_count"})
    with pytest.raises(IncompleteFeaturesError):
        predictor.predict({**ok, "api_calls_count": None})  # unknown is never turned into 0
    for bad in (-1, 2.5, "abc", True):
        with pytest.raises(InvalidFeaturesError):
            predictor.predict({**ok, "api_calls_count": bad})


def test_loading_detects_schema_problems(trained, tmp_path):
    _, _, _, model_dir = trained
    import shutil
    for mutate, message in (
        (lambda s: s.update(feature_names=list(reversed(s["feature_names"]))), "feature list"),
        (lambda s: s.update(extractor_version="9.9.9"), "extractor"),
        (lambda s: s.update(model_sha256="0" * 64), "SHA-256"),
        (lambda s: s.update(schema_version=99), "schema_version"),
    ):
        copy = tmp_path / message.replace(" ", "_")
        shutil.copytree(model_dir, copy)
        schema = json.loads((copy / "feature_schema.json").read_text())
        mutate(schema)
        (copy / "feature_schema.json").write_text(json.dumps(schema))
        with pytest.raises(ModelSchemaError, match=message):
            Predictor.load(copy)
    with pytest.raises(ModelSchemaError, match="Train a model first"):
        Predictor.load(tmp_path / "empty")


# ---------------- metrics ----------------
def test_binary_metrics_exact_arithmetic():
    y_true = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    y_pred = [1, 1, 1, 0, 1, 0, 0, 0, 0, 0]
    m = binary_metrics(y_true, y_pred, [0.9, 0.8, 0.7, 0.4, 0.6, 0.1, 0.2, 0.3, 0.2, 0.1])
    assert m["confusion_matrix"] == {"tn": 5, "fp": 1, "fn": 1, "tp": 3}
    assert m["accuracy"] == pytest.approx(0.8)
    assert m["precision"] == pytest.approx(0.75)
    assert m["recall"] == pytest.approx(0.75)
    assert m["f1"] == pytest.approx(0.75)
    assert m["false_positive_rate"] == pytest.approx(1 / 6)
    assert m["false_negative_rate"] == pytest.approx(0.25)
    assert m["roc_auc"] is not None and m["pr_auc"] is not None


def test_undefined_metrics_are_none_not_zero():
    m = binary_metrics([0, 0, 0], [0, 0, 0])
    assert m["precision"] is None and m["recall"] is None and m["f1"] is None
    assert m["false_positive_rate"] == 0.0 and m["roc_auc"] is None


def test_review_band_splits_decided_and_undecided():
    out = review_band_metrics([1, 0, 1, 0], [0.95, 0.05, 0.5, 0.45], 0.4, 0.6)
    assert out["review_count"] == 2 and out["decided_count"] == 2
    assert out["metrics_on_decided"]["accuracy"] == 1.0


# ---------------- evaluation ----------------
def test_evaluation_report_contents_and_labelling(trained):
    split, _, _, model_dir = trained
    predictor = Predictor.load(model_dir)
    report = build_report(predictor, split.train, split.test, seed=1)
    assert report["synthetic_data"] is True
    assert set(report["baselines"]) == {"majority_class", "logistic_regression"}
    assert report["baselines"]["majority_class"]["recall"] in (0.0, 1.0)
    assert set(report["permutation_importance"]) == set(FEATURE_NAMES)
    text = render_markdown(report)
    assert "NOT evidence of detection quality" in text and "Limits of these results" in text


def test_evaluation_refuses_mismatched_or_leaky_data(trained):
    split, _, _, model_dir = trained
    predictor = Predictor.load(model_dir)
    with pytest.raises(ModelingError, match="fingerprint"):
        build_report(predictor, split.train.iloc[:-5], split.test)
    leaky_test = pd.concat([split.test, split.train.iloc[:1]])
    with pytest.raises(ModelingError, match="share SHA-256"):
        build_report(predictor, split.train, leaky_test)
