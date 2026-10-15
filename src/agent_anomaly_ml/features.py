"""Feature engineering: raw run counters -> model inputs.

Ratios (tokens per LLM call, latency per call, distinct-tool ratio) expose
broken relationships that per-feature thresholds cannot see.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CONTINUOUS = [
    "log_tokens", "log_tool_calls", "log_llm_calls", "max_repeat", "retries",
    "distinct_ratio", "log_tokens_per_llm", "log_latency_per_call",
]
BINARY = ["escalated", "verification_failed"]
FEATURES = CONTINUOUS + BINARY


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    calls = (df["llm_calls"] + df["tool_calls"]).clip(lower=1)
    return pd.DataFrame(
        {
            "log_tokens": np.log1p(df["tokens"]),
            "log_tool_calls": np.log1p(df["tool_calls"]),
            "log_llm_calls": np.log1p(df["llm_calls"]),
            "max_repeat": df["max_repeat"].astype(float),
            "retries": df["retries"].astype(float),
            "distinct_ratio": df["distinct_tools"] / df["tool_calls"].clip(lower=1),
            "log_tokens_per_llm": np.log1p(df["tokens"] / df["llm_calls"].clip(lower=1)),
            "log_latency_per_call": np.log1p(df["latency_s"] / calls),
            "escalated": df["escalated"].astype(float),
            "verification_failed": df["verification_failed"].astype(float),
        },
        index=df.index,
    )[FEATURES]
