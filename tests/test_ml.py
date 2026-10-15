import numpy as np
import pandas as pd
import pytest

from agent_anomaly_ml.data import HARD_TYPES, SOFT_TYPES, make_dataset
from agent_anomaly_ml.evaluate import run_once
from agent_anomaly_ml.models import MahalanobisDetector, fit_detectors


@pytest.fixture(scope="module")
def data():
    return make_dataset(seed=0)


@pytest.fixture(scope="module")
def parts(data):
    return tuple(data[data["split"] == s] for s in ("train", "val", "test"))


@pytest.fixture(scope="module")
def detectors(parts):
    train, val, _ = parts
    return fit_detectors(train, val, seed=0)


def test_dataset_is_deterministic():
    pd.testing.assert_frame_equal(make_dataset(3, 500), make_dataset(3, 500))


def test_training_period_is_known_good(parts):
    assert not parts[0]["is_anomaly"].any()


def test_splits_are_chronological_per_agent(data):
    for _, g in data.groupby("agent"):
        tr, va, te = (g[g["split"] == s]["run_index"] for s in ("train", "val", "test"))
        assert tr.max() < va.min() and va.max() < te.min()


def test_all_anomaly_types_present_in_test(parts):
    assert set(parts[2]["anomaly_type"]) - {""} == set(HARD_TYPES + SOFT_TYPES)


def test_rules_catch_hard_miss_soft_and_stay_quiet(detectors, parts):
    test = parts[2]
    flag = detectors["rules"].flag(test)
    t = test["anomaly_type"].to_numpy()
    assert flag[np.isin(t, HARD_TYPES)].all()
    assert flag[np.isin(t, SOFT_TYPES)].mean() < 0.05
    assert flag[~test["is_anomaly"].to_numpy()].mean() < 0.005


@pytest.mark.parametrize("name", ["robust_z", "mahalanobis", "isolation_forest"])
def test_false_positive_rate_is_calibrated_on_unseen_data(detectors, parts, name):
    test = parts[2]
    fpr = detectors[name].flag(test)[~test["is_anomaly"].to_numpy()].mean()
    assert fpr < 0.05  # target is 2%


def test_calibration_ignores_anomalies_in_validation(parts):
    train, val, _ = parts
    a = MahalanobisDetector().fit(train).calibrate(val[~val["is_anomaly"]])
    b = fit_detectors(train, val)["mahalanobis"]
    assert a.thresholds == b.thresholds


def test_learned_detector_sees_soft_anomalies_rules_cannot(detectors, parts):
    test = parts[2]
    soft = np.isin(test["anomaly_type"].to_numpy(), SOFT_TYPES)
    learned = detectors["mahalanobis"].flag(test)[soft].mean()
    rules = detectors["rules"].flag(test)[soft].mean()
    assert learned > rules + 0.3


def test_hybrid_never_loses_recall_vs_rules():
    res = run_once(seed=1, n_per_agent=2000).set_index("detector")
    assert res.loc["rules+mahalanobis", "recall"] >= res.loc["rules", "recall"]
    assert res.loc["rules+mahalanobis", "recall_hard"] == 1.0


def test_model_roundtrip(tmp_path, parts):
    import joblib

    train, val, test = parts
    det = MahalanobisDetector().fit(train).calibrate(val[~val["is_anomaly"]])
    joblib.dump(det, tmp_path / "m.joblib")
    again = joblib.load(tmp_path / "m.joblib")
    np.testing.assert_allclose(det.score(test), again.score(test))
    assert (det.flag(test) == again.flag(test)).all()
