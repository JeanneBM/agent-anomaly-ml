"""Leakage-free evaluation, repeated over several seeds."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .data import ANOMALY_TYPES, SOFT_TYPES, HARD_TYPES, make_dataset
from .models import fit_detectors


def _metrics(y: np.ndarray, flag: np.ndarray, types: np.ndarray, score: np.ndarray | None):
    tp = int((flag & y).sum())
    fp = int((flag & ~y).sum())
    fn = int((~flag & y).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision == precision and recall == recall and (precision + recall) else float("nan")
    row = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fp / max(int((~y).sum()), 1),
        "avg_precision": average_precision_score(y, score) if score is not None else float("nan"),
    }
    for t in ANOMALY_TYPES:
        m = types == t
        row[f"recall_{t}"] = float(flag[m].mean()) if m.any() else float("nan")
    row["recall_hard"] = float(flag[np.isin(types, HARD_TYPES)].mean())
    row["recall_soft"] = float(flag[np.isin(types, SOFT_TYPES)].mean())
    return row


def run_once(seed: int, n_per_agent: int = 4000, anomaly_rate: float = 0.06, target_fpr: float = 0.02):
    data = make_dataset(seed, n_per_agent, anomaly_rate)
    train, val, test = (data[data["split"] == s] for s in ("train", "val", "test"))
    assert not train["is_anomaly"].any(), "training data must be known-good"
    det = fit_detectors(train, val, seed=seed, target_fpr=target_fpr)

    y = test["is_anomaly"].to_numpy()
    types = test["anomaly_type"].to_numpy()
    flags = {name: d.flag(test) for name, d in det.items()}
    scores = {name: d.score(test) for name, d in det.items()}
    for learned in ("mahalanobis", "isolation_forest"):  # hybrid: hard rules OR learned detector
        flags[f"rules+{learned}"] = flags["rules"] | flags[learned]
        scores[f"rules+{learned}"] = None

    rows = []
    for name, f in flags.items():
        r = _metrics(y, f, types, scores[name] if name != "rules" and "+" not in name else None)
        rows.append({"seed": seed, "detector": name, **r})
    return pd.DataFrame(rows)


def run_experiment(seeds=range(5), **kw) -> pd.DataFrame:
    return pd.concat([run_once(s, **kw) for s in seeds], ignore_index=True)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    cols = ["precision", "recall", "f1", "fpr", "avg_precision", "recall_hard", "recall_soft"]
    cols += [f"recall_{t}" for t in ANOMALY_TYPES]
    g = results.groupby("detector")[cols]
    mean, std = g.mean(), g.std()
    out = mean.copy().astype(object)
    for c in cols:
        out[c] = [
            "-" if np.isnan(m) else (f"{m:.2f} ± {s:.2f}" if not np.isnan(s) else f"{m:.2f}")
            for m, s in zip(mean[c], std[c])
        ]
    order = ["rules", "robust_z", "mahalanobis", "isolation_forest",
             "rules+mahalanobis", "rules+isolation_forest"]
    return out.loc[[o for o in order if o in out.index]]


def to_markdown(summary: pd.DataFrame, seeds: int, n_per_agent: int, rate: float, fpr: float) -> str:
    lines = [
        f"# Evaluation ({seeds} seeds, {n_per_agent} runs/agent, anomaly rate {rate:.0%}, target FPR {fpr:.0%})",
        "",
        "Mean ± std across seeds, on the held-out test period. Thresholds are calibrated on "
        "normal validation runs only.",
        "",
    ]
    cols = list(summary.columns)
    lines.append("| detector | " + " | ".join(cols) + " |")
    lines.append("|---|" + "---|" * len(cols))
    for name, row in summary.iterrows():
        lines.append(f"| {name} | " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"
