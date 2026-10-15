"""Synthetic run-level telemetry for three agent types, with labelled anomalies.

Design choices that matter for honest evaluation:
  * Time split: train (first 60%) -> validation (next 20%) -> test (last 20%), per agent.
  * The training period is anomaly-free ("known good"). Anomalies appear only in
    validation and test, as they would after a regression is introduced.
  * Anomaly types are split into HARD (violate a fixed limit, a rule can catch them)
    and SOFT (stay within every fixed limit, only a learned baseline can notice).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

AGENT_PROFILES = {
    "diagnostic": dict(llm=4, tools=6, tok_per_llm=1400, lat_llm=2.2, lat_tool=0.7, esc=0.08, vfail=0.00),
    "remediation": dict(llm=5, tools=5, tok_per_llm=1800, lat_llm=2.5, lat_tool=1.4, esc=0.12, vfail=0.04),
    "rag": dict(llm=3, tools=2, tok_per_llm=2200, lat_llm=3.0, lat_tool=0.5, esc=0.06, vfail=0.00),
}
PRICE_PER_TOKEN = 2.5e-5
HARD_LIMITS = dict(tokens=30_000, cost_usd=0.75, max_repeat=3, tool_calls=40)

HARD_TYPES = ("hard_loop", "hard_budget")
SOFT_TYPES = ("soft_drift", "soft_pattern")
ANOMALY_TYPES = HARD_TYPES + SOFT_TYPES

SPLITS = (("train", 0.0, 0.6), ("val", 0.6, 0.8), ("test", 0.8, 1.0))


def _normal_runs(rng: np.random.Generator, agent: str, n: int) -> pd.DataFrame:
    p = AGENT_PROFILES[agent]
    llm = np.minimum(1 + rng.poisson(p["llm"] - 1, n), 12)
    tools = np.minimum(1 + rng.poisson(p["tools"] - 1, n), 20)
    tokens = np.rint(llm * rng.lognormal(np.log(p["tok_per_llm"]), 0.25, n)).astype(int)
    distinct = 1 + rng.binomial(tools - 1, 0.6)
    max_repeat = np.minimum(1 + rng.binomial(2, 0.2, n), tools)
    retries = rng.poisson(0.25, n)
    latency = (
        llm * p["lat_llm"] * rng.lognormal(0, 0.2, n)
        + tools * p["lat_tool"] * rng.lognormal(0, 0.25, n)
        + retries * 1.5
    )
    return pd.DataFrame(
        {
            "agent": agent,
            "llm_calls": llm,
            "tool_calls": tools,
            "distinct_tools": distinct,
            "max_repeat": max_repeat,
            "retries": retries,
            "tokens": tokens,
            "cost_usd": tokens * PRICE_PER_TOKEN * rng.lognormal(0, 0.05, n),
            "latency_s": latency,
            "escalated": (rng.random(n) < p["esc"]).astype(int),
            "verification_failed": (rng.random(n) < p["vfail"]).astype(int),
        }
    )


def _sync_cost(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df["cost_usd"] = df["tokens"] * PRICE_PER_TOKEN * rng.lognormal(0, 0.05, len(df))
    return df


def _inject(rng: np.random.Generator, df: pd.DataFrame, kind: str) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    lim = HARD_LIMITS
    if kind == "hard_loop":  # same call repeated over and over
        rep = rng.integers(6, 16, n)
        df["max_repeat"] = rep
        df["tool_calls"] = df["tool_calls"] + rep * 2
        df["tokens"] = (df["tokens"] * rng.uniform(1.3, 2.0, n)).astype(int)
        df["latency_s"] = df["latency_s"] * rng.uniform(1.5, 3.0, n)
    elif kind == "hard_budget":  # runaway token use
        df["tokens"] = (lim["tokens"] * rng.uniform(1.1, 2.0, n)).astype(int)
        df["llm_calls"] = df["llm_calls"] + rng.integers(6, 12, n)
        df["latency_s"] = df["latency_s"] * rng.uniform(1.5, 2.5, n)
    elif kind == "soft_drift":  # everything gets 1.5-2x heavier, still under every limit
        df["llm_calls"] = np.rint(df["llm_calls"] * rng.uniform(1.2, 1.6, n)).astype(int)
        df["tool_calls"] = np.rint(df["tool_calls"] * rng.uniform(1.4, 1.9, n)).astype(int)
        df["tokens"] = np.minimum(
            (df["tokens"] * rng.uniform(1.5, 2.1, n)).astype(int), int(0.95 * lim["tokens"])
        )
        df["latency_s"] = df["latency_s"] * rng.uniform(1.5, 2.2, n)
    elif kind == "soft_pattern":  # totals look fine, relationships between them break
        variant = rng.integers(0, 3, n)
        a, b, c = variant == 0, variant == 1, variant == 2
        # a) near-empty LLM responses: many calls, very few tokens
        df.loc[a, "tokens"] = (df.loc[a, "llm_calls"] * rng.uniform(150, 400, a.sum())).astype(int)
        # b) tool calls that return instantly: calls logged but barely any time spent
        df.loc[b, "latency_s"] = (
            df.loc[b, "llm_calls"] * 1.0 + df.loc[b, "tool_calls"] * rng.uniform(0.01, 0.05, b.sum())
        )
        # c) retry storm hidden inside otherwise normal totals
        df.loc[c, "retries"] = rng.integers(2, 5, c.sum())
        df.loc[c, "distinct_tools"] = df.loc[c, "tool_calls"]
    else:
        raise ValueError(kind)
    df["distinct_tools"] = np.minimum(df["distinct_tools"], df["tool_calls"])
    return _sync_cost(df, rng)


def make_dataset(seed: int = 0, n_per_agent: int = 4000, anomaly_rate: float = 0.06) -> pd.DataFrame:
    """Return one row per agent run with `split`, `is_anomaly` and `anomaly_type` labels."""
    rng = np.random.default_rng(seed)
    parts = []
    for agent in AGENT_PROFILES:
        df = _normal_runs(rng, agent, n_per_agent)
        df["run_index"] = np.arange(n_per_agent)
        df["anomaly_type"] = ""
        df["split"] = "train"
        for name, lo, hi in SPLITS:
            idx = np.arange(int(lo * n_per_agent), int(hi * n_per_agent))
            df.loc[idx, "split"] = name
        later = np.flatnonzero(df["split"] != "train")
        chosen = later[rng.random(len(later)) < anomaly_rate]
        kinds = rng.choice(ANOMALY_TYPES, size=len(chosen))
        for kind in ANOMALY_TYPES:
            rows = chosen[kinds == kind]
            if len(rows):
                df.iloc[rows, df.columns.get_indexer(df.columns)] = _inject(rng, df.iloc[rows], kind)
                df.iloc[rows, df.columns.get_loc("anomaly_type")] = kind
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    out["is_anomaly"] = out["anomaly_type"] != ""
    return out
