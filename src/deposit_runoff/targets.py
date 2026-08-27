from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .features import _eligible_snapshot_rows, validate_daily_panel


def build_v2_robustness_snapshot(
    daily: pd.DataFrame,
    snapshot_date: str | pd.Timestamp,
    *,
    reference_days: int = 30,
    observation_days: int = 30,
    lead_days: int = 14,
    outcome_days: int = 30,
    stable_ratio: float = 0.90,
    runoff_ratio: float = 0.50,
    materiality_floor_units: float | None = None,
) -> pd.DataFrame:
    """Construct the stricter V2 robustness cohort for one snapshot.

    Window layout:
        reference:   t-59 ... t-30
        observation: t-29 ... t
        lead:        t+1  ... t+14
        outcome:     t+15 ... t+44

    Important: the lead-window screen uses future behavior and therefore defines a
    *conditional evaluation cohort*. It is useful for model-validity analysis but is
    not a deployable population-selection rule available at scoring time.
    """
    if reference_days != 30 or observation_days != 30 or lead_days != 14 or outcome_days != 30:
        raise ValueError("Public reconstruction currently implements the verified 30/30/14/30 design")

    daily = validate_daily_panel(daily)
    t = pd.Timestamp(snapshot_date)
    cohort = _eligible_snapshot_rows(daily, t)
    if cohort.empty:
        return pd.DataFrame()
    ids = set(cohort["account_id"])

    ref_start, ref_end = t - pd.Timedelta(days=59), t - pd.Timedelta(days=30)
    obs_start, obs_end = t - pd.Timedelta(days=29), t
    recent_start = t - pd.Timedelta(days=6)
    lead_start, lead_end = t + pd.Timedelta(days=1), t + pd.Timedelta(days=14)
    out_start, out_end = t + pd.Timedelta(days=15), t + pd.Timedelta(days=44)

    panel = daily.loc[daily["account_id"].isin(ids)].copy()
    ref = panel.loc[panel["date"].between(ref_start, ref_end)]
    obs = panel.loc[panel["date"].between(obs_start, obs_end)]
    recent = panel.loc[panel["date"].between(recent_start, obs_end)]
    lead = panel.loc[panel["date"].between(lead_start, lead_end)]
    outcome = panel.loc[panel["date"].between(out_start, out_end)]

    ref_g = ref.groupby("account_id", observed=True)["balance"].agg(
        reference_balance="median", n_ref="count"
    )
    obs_g = obs.groupby("account_id", observed=True)["balance"].agg(
        avg_bal_30d="mean",
        bal_std_30d="std",
        min_bal_30d="min",
        max_bal_30d="max",
        n_obs_30d="count",
    )
    recent_g = recent.groupby("account_id", observed=True)["balance"].mean().rename("avg_bal_7d")
    out_g = outcome.groupby("account_id", observed=True)["balance"].agg(
        outcome_avg="mean", n_outcome="count"
    )

    lead = lead.sort_values(["account_id", "date"])
    lead["lead_rolling7"] = (
        lead.groupby("account_id", observed=True)["balance"]
        .rolling(7, min_periods=7)
        .mean()
        .reset_index(level=0, drop=True)
    )
    lead_g = lead.groupby("account_id", observed=True).agg(
        lead7_min=("lead_rolling7", "min"), n_lead=("balance", "count")
    )

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
        base.merge(ref_g, on="account_id", how="inner", validate="one_to_one")
        .merge(obs_g, on="account_id", how="inner", validate="one_to_one")
        .merge(recent_g, on="account_id", how="inner", validate="one_to_one")
        .merge(lead_g, on="account_id", how="inner", validate="one_to_one")
        .merge(out_g, on="account_id", how="inner", validate="one_to_one")
    )

    floor = 0.0 if materiality_floor_units is None else float(materiality_floor_units)
    keep = (
        (out["reference_balance"] >= floor)
        & (out["avg_bal_7d"] >= stable_ratio * out["reference_balance"])
        & (out["lead7_min"] > runoff_ratio * out["reference_balance"])
        & (out["n_ref"] == 30)
        & (out["n_obs_30d"] == 30)
        & (out["n_lead"] == 14)
        & (out["n_outcome"] == 30)
    )
    out = out.loc[keep].copy()

    denom = out["avg_bal_30d"].abs().replace(0, np.nan)
    out["snapshot_date"] = t
    out["asof_vs_30d_avg_pct"] = (out["asof_balance"] - out["avg_bal_30d"]) / denom
    out["recent7_vs_30d_pct"] = (out["avg_bal_7d"] - out["avg_bal_30d"]) / denom
    out["bal_volatility_cv_30d"] = out["bal_std_30d"] / denom
    out["bal_range_pct_30d"] = (out["max_bal_30d"] - out["min_bal_30d"]) / denom
    out["account_tenure_days"] = (t - out["open_date"]).dt.days
    out["runoff_flag"] = (
        out["outcome_avg"] <= runoff_ratio * out["reference_balance"]
    ).astype("int8")
    out["runoff_amount"] = (
        out["reference_balance"] - out["outcome_avg"]
    ).clip(lower=0)

    return out.drop(columns=["open_date", "outcome_avg"])


def build_v2_robustness_dataset(
    daily: pd.DataFrame,
    snapshot_dates: Iterable[str | pd.Timestamp],
    **kwargs,
) -> pd.DataFrame:
    """Build the multi-snapshot V2 robustness dataset."""
    parts = [build_v2_robustness_snapshot(daily, d, **kwargs) for d in snapshot_dates]
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
