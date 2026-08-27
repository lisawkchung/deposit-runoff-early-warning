from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def chronological_split(
    frame: pd.DataFrame,
    *,
    train_dates: Iterable[str | pd.Timestamp],
    validation_date: str | pd.Timestamp,
    test_date: str | pd.Timestamp,
    snapshot_col: str = "snapshot_date",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create a forward-only train/validation/test split by snapshot date."""
    x = frame.copy()
    x[snapshot_col] = pd.to_datetime(x[snapshot_col])
    train_dates = pd.to_datetime(list(train_dates))
    val_date = pd.Timestamp(validation_date)
    tst_date = pd.Timestamp(test_date)

    if not (max(train_dates) < val_date < tst_date):
        raise ValueError("Chronological split must satisfy max(train) < validation < test")

    train = x.loc[x[snapshot_col].isin(train_dates)].copy()
    val = x.loc[x[snapshot_col] == val_date].copy()
    test = x.loc[x[snapshot_col] == tst_date].copy()
    if train.empty or val.empty or test.empty:
        raise ValueError("Chronological split produced an empty partition")
    return train, val, test
