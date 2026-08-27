from __future__ import annotations

import numpy as np
import pandas as pd


def top_fraction_metrics(
    frame: pd.DataFrame,
    score: pd.Series | np.ndarray,
    *,
    fraction: float = 0.10,
    target_col: str = "runoff_flag",
    severity_col: str = "runoff_amount",
) -> dict[str, float]:
    """Measure precision, event capture, and labeled-severity capture in a top slice."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    x = frame.copy()
    x["_score"] = np.asarray(score)
    x["_labeled_severity"] = np.where(x[target_col].eq(1), x[severity_col], 0.0)
    x = x.sort_values("_score", ascending=False).reset_index(drop=True)
    n_top = max(1, int(len(x) * fraction))
    top = x.iloc[:n_top]

    total_events = x[target_col].sum()
    total_severity = x["_labeled_severity"].sum()
    return {
        "precision": float(top[target_col].mean()),
        "event_capture": float(top[target_col].sum() / total_events) if total_events else np.nan,
        "severity_capture": (
            float(top["_labeled_severity"].sum() / total_severity) if total_severity else np.nan
        ),
    }


def balance_extreme_capture(
    frame: pd.DataFrame,
    *,
    largest_first: bool,
    fraction: float = 0.10,
    balance_col: str = "avg_bal_30d",
) -> dict[str, float]:
    """Use balance alone as a ranking score for the same capture metrics."""
    score = frame[balance_col].astype(float)
    if not largest_first:
        score = -score
    return top_fraction_metrics(frame, score, fraction=fraction)
