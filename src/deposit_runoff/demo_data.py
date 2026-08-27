from __future__ import annotations

import numpy as np
import pandas as pd


def make_demo_daily_panel(
    *,
    n_accounts: int = 800,
    start_date: str = "2023-12-01",
    end_date: str = "2024-07-30",
    seed: int = 42,
) -> pd.DataFrame:
    """Create deterministic synthetic daily balances for public code-path reproduction."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, end_date, freq="D")
    account_ids = np.array([f"A{i:05d}" for i in range(n_accounts)])

    account_type = rng.choice(["checking", "savings"], n_accounts, p=[0.65, 0.35])
    product_group = np.where(
        account_type == "checking",
        rng.choice(["Checking_A", "Checking_B", "Checking_C"], n_accounts),
        rng.choice(["Savings_A", "Savings_B"], n_accounts),
    )
    base_balance = rng.lognormal(mean=8.0, sigma=1.15, size=n_accounts)
    deposit_rate = np.where(
        account_type == "savings",
        rng.choice([0.02, 0.03, 0.04], n_accounts),
        rng.choice([0.00, 0.001, 0.01], n_accounts),
    ).astype(float)
    deposit_rate[rng.random(n_accounts) < 0.10] = np.nan
    open_date = pd.Timestamp(start_date) - pd.to_timedelta(rng.integers(90, 2500, n_accounts), unit="D")

    event_dates = np.full(n_accounts, np.datetime64("NaT"), dtype="datetime64[ns]")
    event_mask = rng.random(n_accounts) < 0.28
    eligible_event_dates = pd.date_range("2024-02-10", "2024-07-05", freq="D")
    event_dates[event_mask] = rng.choice(eligible_event_dates.values, event_mask.sum())
    severe = rng.uniform(0.10, 0.45, n_accounts)
    gradual = rng.random(n_accounts) < 0.55

    rows = []
    for i, account_id in enumerate(account_ids):
        noise = rng.normal(0, 0.035, len(dates))
        seasonal = 1 + 0.03 * np.sin(np.arange(len(dates)) / 7 * 2 * np.pi)
        bal = np.maximum(base_balance[i] * seasonal * (1 + noise), 0.01)
        if not np.isnat(event_dates[i]):
            event = pd.Timestamp(event_dates[i])
            idx = np.searchsorted(dates.values, np.datetime64(event))
            if gradual[i]:
                ramp_start = max(0, idx - 14)
                ramp = np.linspace(1.0, max(severe[i], 0.55), idx - ramp_start + 1)
                bal[ramp_start : idx + 1] *= ramp
            bal[idx:] *= severe[i]

        frame = pd.DataFrame(
            {
                "date": dates,
                "account_id": account_id,
                "account_type": account_type[i],
                "product_group": product_group[i],
                "balance": bal,
                "deposit_rate": deposit_rate[i],
                "open_date": open_date[i],
                "status": "ACTIVE",
                "is_internal": False,
            }
        )
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)
