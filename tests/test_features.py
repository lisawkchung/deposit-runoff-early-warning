import pandas as pd
import pytest

from deposit_runoff.features import build_v1_snapshot, validate_daily_panel


def test_duplicate_account_day_is_rejected(simple_panel):
    bad = pd.concat([simple_panel, simple_panel.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate account_id/date"):
        validate_daily_panel(bad)


def test_snapshot_is_one_row_per_account(simple_panel):
    out = build_v1_snapshot(simple_panel, "2024-01-31")
    assert not out["account_id"].duplicated().any()
    assert len(out) == 2
