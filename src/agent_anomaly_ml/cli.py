from __future__ import annotations

import argparse
import sys

import joblib
import pandas as pd

from .data import make_dataset
from .evaluate import run_experiment, summarize, to_markdown
from .models import IForestDetector, MahalanobisDetector, RobustZDetector

DETECTORS = {
    "mahalanobis": MahalanobisDetector,
    "isolation_forest": IForestDetector,
    "robust_z": RobustZDetector,
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="agent-anomaly-ml")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="write a synthetic dataset to CSV")
    g.add_argument("--out", required=True)
    g.add_argument("--seed", type=int, default=0)

    r = sub.add_parser("run", help="run the evaluation experiment")
    r.add_argument("--seeds", type=int, default=5)
    r.add_argument("--runs-per-agent", type=int, default=4000)
    r.add_argument("--anomaly-rate", type=float, default=0.06)
    r.add_argument("--fpr", type=float, default=0.02)
    r.add_argument("--out", help="write the markdown report here")

    t = sub.add_parser("train", help="fit + calibrate a detector on a dataset CSV")
    t.add_argument("data")
    t.add_argument("--detector", choices=sorted(DETECTORS), default="mahalanobis")
    t.add_argument("--out", required=True)
    t.add_argument("--fpr", type=float, default=0.02)

    s = sub.add_parser("score", help="flag runs in a CSV using a saved model")
    s.add_argument("data")
    s.add_argument("--model", required=True)

    a = p.parse_args(argv)

    if a.cmd == "generate":
        make_dataset(a.seed).to_csv(a.out, index=False)
    elif a.cmd == "run":
        res = run_experiment(range(a.seeds), n_per_agent=a.runs_per_agent,
                             anomaly_rate=a.anomaly_rate, target_fpr=a.fpr)
        md = to_markdown(summarize(res), a.seeds, a.runs_per_agent, a.anomaly_rate, a.fpr)
        print(md)
        if a.out:
            open(a.out, "w", encoding="utf-8").write(md)
    elif a.cmd == "train":
        df = pd.read_csv(a.data)
        df["is_anomaly"] = df["is_anomaly"].astype(bool)
        train = df[df["split"] == "train"]
        val_normal = df[(df["split"] == "val") & ~df["is_anomaly"]]
        det = DETECTORS[a.detector](target_fpr=a.fpr).fit(train).calibrate(val_normal)
        joblib.dump(det, a.out)
    elif a.cmd == "score":
        det = joblib.load(a.model)
        df = pd.read_csv(a.data)
        df["anomaly_score"] = det.score(df)
        df["flagged"] = det.flag(df)
        print(df.loc[df["flagged"], ["agent", "run_index", "anomaly_score"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
