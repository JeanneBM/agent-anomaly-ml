"""Detectors. All share one interface: fit on known-good runs, score, calibrate, flag.

  RuleDetector     fixed limits (what a threshold-only monitor does)
  RobustZDetector  per-agent median/MAD, max |z| over continuous features
  MahalanobisDetector  per-agent covariance distance: sees broken feature relationships
  IForestDetector  per-agent Isolation Forest over all features (multivariate)

Thresholds are calibrated on *normal validation runs only* to a target false
positive rate. The test set is never used for fitting or calibration.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import IsolationForest

from .data import HARD_LIMITS
from .features import CONTINUOUS, feature_frame


class RuleDetector:
    name = "rules"

    def score(self, df: pd.DataFrame) -> np.ndarray:
        lim = HARD_LIMITS
        hit = (
            (df["tokens"] > lim["tokens"])
            | (df["cost_usd"] > lim["cost_usd"])
            | (df["max_repeat"] > lim["max_repeat"])
            | (df["tool_calls"] > lim["tool_calls"])
        )
        return hit.to_numpy(float)

    def flag(self, df: pd.DataFrame) -> np.ndarray:
        return self.score(df) > 0.5


@dataclass
class _PerAgentDetector:
    seed: int = 0
    target_fpr: float = 0.02
    models: dict = field(default_factory=dict)
    thresholds: dict = field(default_factory=dict)

    name = "base"

    def _fit_one(self, X: pd.DataFrame):
        raise NotImplementedError

    def _score_one(self, model, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def fit(self, train: pd.DataFrame) -> "_PerAgentDetector":
        for agent, part in train.groupby("agent"):
            self.models[agent] = self._fit_one(feature_frame(part))
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        out = np.zeros(len(df))
        for agent, part in df.groupby("agent"):
            pos = df.index.get_indexer(part.index)
            out[pos] = self._score_one(self.models[agent], feature_frame(part))
        return out

    def calibrate(self, val_normal: pd.DataFrame) -> "_PerAgentDetector":
        """Threshold = (1 - target_fpr) quantile of scores on known-good validation runs."""
        for agent, part in val_normal.groupby("agent"):
            self.thresholds[agent] = float(np.quantile(self.score(part), 1 - self.target_fpr))
        return self

    def flag(self, df: pd.DataFrame) -> np.ndarray:
        s = self.score(df)
        thr = df["agent"].map(self.thresholds).to_numpy(float)
        return s > thr


class RobustZDetector(_PerAgentDetector):
    name = "robust_z"

    def _fit_one(self, X):
        C = X[CONTINUOUS]
        med = C.median()
        mad = (C - med).abs().median() * 1.4826
        # Discrete features (e.g. max_repeat) can have MAD == 0; fall back to the std.
        scale = mad.where(mad > 1e-3, C.std()).clip(lower=1e-3)
        return med, scale

    def _score_one(self, model, X):
        med, mad = model
        z = ((X[CONTINUOUS] - med) / mad).abs()
        return z.max(axis=1).to_numpy()


class MahalanobisDetector(_PerAgentDetector):
    name = "mahalanobis"

    def _fit_one(self, X):
        return EmpiricalCovariance().fit(X[CONTINUOUS].to_numpy())

    def _score_one(self, model, X):
        return model.mahalanobis(X[CONTINUOUS].to_numpy())


class IForestDetector(_PerAgentDetector):
    name = "isolation_forest"

    def _fit_one(self, X):
        return IsolationForest(n_estimators=300, random_state=self.seed).fit(X.to_numpy())

    def _score_one(self, model, X):
        return -model.score_samples(X.to_numpy())  # higher = more anomalous


def fit_detectors(train: pd.DataFrame, val: pd.DataFrame, seed: int = 0, target_fpr: float = 0.02):
    """Fit on train (known good), calibrate on normal validation runs. Never sees test."""
    val_normal = val[~val["is_anomaly"]]
    z = RobustZDetector(seed=seed, target_fpr=target_fpr).fit(train).calibrate(val_normal)
    mah = MahalanobisDetector(seed=seed, target_fpr=target_fpr).fit(train).calibrate(val_normal)
    iso = IForestDetector(seed=seed, target_fpr=target_fpr).fit(train).calibrate(val_normal)
    return {"rules": RuleDetector(), "robust_z": z, "mahalanobis": mah, "isolation_forest": iso}
