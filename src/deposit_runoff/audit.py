from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .metrics import top_fraction_metrics


def deterioration_sensitivity(
    frame: pd.DataFrame,
    pred: np.ndarray,
    thresholds: tuple[float, ...] = (0.10, 0.25, 0.50),
) -> pd.DataFrame:
    """Post-hoc robustness check after excluding already-deteriorated accounts.

    `threshold=0.10` keeps accounts whose as-of balance is no more than 10% below
    their recent 30-day average.
    """
    rows = []
    for threshold in thresholds:
        keep = frame["asof_vs_30d_avg_pct"] > -threshold
        cohort = frame.loc[keep]
        cohort_pred = np.asarray(pred)[keep.to_numpy()]
        top = top_fraction_metrics(cohort, cohort_pred, fraction=0.10)
        rows.append(
            {
                "max_existing_decline": threshold,
                "n": len(cohort),
                "prevalence": cohort["runoff_flag"].mean(),
                "roc_auc": roc_auc_score(cohort["runoff_flag"], cohort_pred),
                "pr_auc": average_precision_score(cohort["runoff_flag"], cohort_pred),
                "top10_precision": top["precision"],
                "top10_event_capture": top["event_capture"],
            }
        )
    return pd.DataFrame(rows)
