from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

REQUIRED_DAILY_COLUMNS = {
    "date",
    "account_id",
    "account_type",
    "product_group",
    "balance",
    "deposit_rate",
    "open_date",
    "status",
    "is_internal",
}


def validate_daily_panel(daily: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the generic account-by-day input panel.

    A duplicate account/date pair is treated as a hard error because it can silently
    multiply balances during feature joins.
    """
    missing = REQUIRED_DAILY_COLUMNS - set(daily.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = daily.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["open_date"] = pd.to_datetime(out["open_date"])

    dup = out.duplicated(["account_id", "date"])
    if dup.any():
        raise ValueError("Duplicate account_id/date rows would corrupt point-in-time joins")
    return out


def _eligible_snapshot_rows(daily: pd.DataFrame, snapshot: pd.Timestamp) -> pd.DataFrame:
    cohort = daily.loc[
        (daily["date"] == snapshot)
        & (daily["status"] == "ACTIVE")
        & (daily["balance"] > 0)
        & (daily["account_type"].isin(["checking", "savings"]))
        & (~daily["is_internal"].astype(bool))
    ].copy()
    return cohort


def build_v1_snapshot(
    daily: pd.DataFrame,
    snapshot_date: str | pd.Timestamp,
    *,
    history_days: int = 30,
    recent_days: int = 7,
    future_days: int = 30,
    runoff_ratio: float = 0.50,
) -> pd.DataFrame:
    """Build one point-in-time V1 account snapshot.

    Features use dates <= snapshot. The target uses only dates after the snapshot.
    This separation is the core leakage guardrail of the initial model.
    """
    daily = validate_daily_panel(daily)
    snapshot = pd.Timestamp(snapshot_date)
    cohort = _eligible_snapshot_rows(daily, snapshot)
    if cohort.empty:
        return pd.DataFrame()

    ids = set(cohort["account_id"])
    hist_start = snapshot - pd.Timedelta(days=history_days - 1)
    recent_start = snapshot - pd.Timedelta(days=recent_days - 1)
    future_start = snapshot + pd.Timedelta(days=1)
    future_end = snapshot + pd.Timedelta(days=future_days)

    hist = daily.loc[
        daily["account_id"].isin(ids)
        & daily["date"].between(hist_start, snapshot)
    ]
    recent = hist.loc[hist["date"].between(recent_start, snapshot)]
    future = daily.loc[
        daily["account_id"].isin(ids)
        & daily["date"].between(future_start, future_end)
    ]

    h = hist.groupby("account_id", observed=True)["balance"].agg(
        avg_bal_30d="mean",
        bal_std_30d="std",
        min_bal_30d="min",
        max_bal_30d="max",
        n_obs_30d="count",
    )
    r = recent.groupby("account_id", observed=True)["balance"].mean().rename("avg_bal_7d")
    f = future.groupby("account_id", observed=True)["balance"].mean().rename("avg_bal_future_30d")

    base = cohort[
        [
            "account_id",
            "account_type",
            "product_group",
            "balance",
            "deposit_rate",
            "open_date",
        ]
    ].rename(columns={"balance": "asof_balance"})

    out = (
        base.merge(h, on="account_id", how="inner", validate="one_to_one")
        .merge(r, on="account_id", how="inner", validate="one_to_one")
        .merge(f, on="account_id", how="inner", validate="one_to_one")
    )
    out = out.loc[(out["avg_bal_30d"] > 0) & out["avg_bal_future_30d"].notna()].copy()

    denom = out["avg_bal_30d"].abs().replace(0, np.nan)
    out["snapshot_date"] = snapshot
    out["asof_vs_30d_avg_pct"] = (out["asof_balance"] - out["avg_bal_30d"]) / denom
    out["recent7_vs_30d_pct"] = (out["avg_bal_7d"] - out["avg_bal_30d"]) / denom
    out["bal_volatility_cv_30d"] = out["bal_std_30d"] / denom
    out["bal_range_pct_30d"] = (out["max_bal_30d"] - out["min_bal_30d"]) / denom
    out["account_tenure_days"] = (snapshot - out["open_date"]).dt.days
    out["runoff_flag"] = (
        out["avg_bal_future_30d"] <= runoff_ratio * out["avg_bal_30d"]
    ).astype("int8")
    out["runoff_amount"] = (out["avg_bal_30d"] - out["avg_bal_future_30d"]).clip(lower=0)

    return out.drop(columns=["avg_bal_future_30d", "open_date"])


def build_v1_dataset(
    daily: pd.DataFrame,
    snapshot_dates: Iterable[str | pd.Timestamp],
    **kwargs,
) -> pd.DataFrame:
    """Build and concatenate V1 snapshots."""
    parts = [build_v1_snapshot(daily, d, **kwargs) for d in snapshot_dates]
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
