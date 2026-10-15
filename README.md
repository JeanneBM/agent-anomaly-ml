# Agent Anomaly ML — PoC

Learned anomaly detection for AI-agent run telemetry. It complements the deterministic rules of a read-only agent monitor: rules catch hard violations (a tool called 12 times in a row, a runaway token budget); a learned baseline can notice runs that stay inside every fixed limit but no longer look like normal behaviour.

**Question this project answers:** *what do learned detectors add on top of threshold rules, and at what cost in false alarms?*

## Setup

- **Data:** synthetic run-level telemetry for three agent types (diagnostic, remediation, RAG): LLM calls, tool calls, repeats, retries, tokens, cost, latency, escalation, failed verification.
- **Time split per agent:** train (first 60%, anomaly-free, "known good") → validation (20%) → test (20%). Anomalies appear only in validation and test, as after a regression.
- **Anomalies (≈6% of later runs):**
  - *hard* — `hard_loop` (same call repeated), `hard_budget` (runaway tokens). They break a fixed limit.
  - *soft* — `soft_drift` (everything 1.5–2x heavier but under every limit), `soft_pattern` (near-empty LLM responses, instant tool calls, hidden retry storms). Totals look normal; relationships break.
- **Detectors**
  - `rules` — fixed limits, like a threshold-only monitor.
  - `robust_z` — per-agent median/MAD, max |z| over features.
  - `mahalanobis` — per-agent covariance distance; sees broken feature relationships.
  - `isolation_forest` — per-agent Isolation Forest.
  - `rules+…` — hybrid: hard rules OR learned detector.
- **No leakage:** detectors are fit on train; alert thresholds are set on *normal validation runs only* to a 2% target false-positive rate; the test period is only scored. Reported as mean ± std over 5 seeds.

## Results (5 seeds, 4000 runs/agent)

| detector | precision | recall | F1 | FPR | recall hard | recall soft |
|---|---|---|---|---|---|---|
| rules | 0.99 ± 0.02 | 0.49 ± 0.02 | 0.65 ± 0.02 | 0.00 | 1.00 | 0.00 |
| robust_z | 0.76 ± 0.10 | 0.53 ± 0.01 | 0.62 ± 0.03 | 0.01 | 0.68 | 0.38 |
| mahalanobis | 0.70 ± 0.05 | 0.76 ± 0.03 | 0.73 ± 0.02 | 0.02 | 0.95 | 0.56 |
| isolation_forest | 0.57 ± 0.02 | 0.44 ± 0.08 | 0.50 ± 0.05 | 0.02 | 0.72 | 0.18 |
| **rules + mahalanobis** | 0.70 ± 0.05 | **0.78 ± 0.02** | **0.74 ± 0.02** | 0.02 | 1.00 | 0.57 |
| rules + isolation_forest | 0.64 ± 0.04 | 0.58 ± 0.02 | 0.61 ± 0.01 | 0.02 | 1.00 | 0.18 |

Full table with per-anomaly-type recall: `agent-anomaly-ml run --out report.md`.

### What the numbers say

- Rules are precise and catch every hard anomaly, but are blind to soft ones (recall ≈ 0). That is expected by construction.
- A learned multivariate baseline recovers more than half of the soft anomalies, mostly the *broken-relationship* kind (`soft_pattern`, ≈ 0.9 recall for Mahalanobis).
- The price is false alarms: ~2% of normal runs are flagged (the calibrated target), which drops precision from 0.99 to ≈ 0.70.
- **Hybrid is the sensible design:** rules guarantee the hard cases, the learned detector adds soft ones.
- `soft_drift` (a uniform 1.5–2x shift) stays hard for every detector at this FPR (best ≈ 0.2 recall). Per-run scoring is the wrong tool for it; it needs window-level drift detection on aggregates.
- Isolation Forest is the weakest here. Trees cannot extrapolate beyond the training range, so far-out values are not scored as more extreme than the most extreme training run.

## Limitations — read before drawing conclusions

- **Synthetic data.** Features are roughly Gaussian-ish and correlated by construction, which favours Mahalanobis. Real agent telemetry is messier; do not expect these numbers to transfer.
- Anomaly types were designed by me, so they reflect my assumptions about failure modes.
- Calibration assumes a reviewed, known-good validation period. Without one, the FPR guarantee does not hold.
- No concept-drift handling or retraining policy; per-agent models are fit once.
- Not connected to real agents yet. Next step: feed it features computed from the shared event schema of the agent monitor (`agent-monitor`), and score real runs.

## Usage

```bash
pip install -e ".[dev]"
pytest

agent-anomaly-ml run --seeds 5 --out report.md      # evaluation experiment
agent-anomaly-ml generate --out data.csv --seed 1   # synthetic dataset
agent-anomaly-ml train data.csv --out model.joblib --detector mahalanobis
agent-anomaly-ml score data.csv --model model.joblib
```

## Layout

```
src/agent_anomaly_ml/
  data.py       synthetic telemetry, anomaly injection, time split
  features.py   counters -> ratios and log features
  models.py     rules, robust z, Mahalanobis, Isolation Forest; calibration
  evaluate.py   multi-seed, leakage-free evaluation and report
  cli.py        generate | run | train | score
tests/          determinism, no-leakage, calibration, hybrid-vs-rules checks
```
