import pytest
import pandas as pd

from deposit_runoff.features import build_v1_snapshot
from deposit_runoff.targets import build_v2_robustness_snapshot


def test_v1_target_uses_future_average(simple_panel):
    out = build_v1_snapshot(simple_panel, "2024-01-31")
    labels = out.set_index("account_id")["runoff_flag"].to_dict()
    assert labels == {"A": 1, "B": 0}


def test_future_changes_label_but_not_features(simple_panel):
    left = build_v1_snapshot(simple_panel, "2024-01-31").set_index("account_id")
    changed = simple_panel.copy()
    changed.loc[
        (changed["account_id"] == "A") & (changed["date"] > pd.Timestamp("2024-01-31")),
        "balance",
    ] = 90.0
    right = build_v1_snapshot(changed, "2024-01-31").set_index("account_id")

    feature_cols = ["asof_balance", "avg_bal_7d", "avg_bal_30d", "recent7_vs_30d_pct"]
    pd.testing.assert_series_equal(left.loc["A", feature_cols], right.loc["A", feature_cols])
    assert left.loc["A", "runoff_flag"] != right.loc["A", "runoff_flag"]


def test_incomplete_future_window_excluded(simple_panel):
    # Truncate panel so account B has only 15 future days after 2024-01-31
    # (data ends 2024-02-15 for B instead of 2024-03-01)
    truncated = simple_panel.loc[
        ~((simple_panel["account_id"] == "B") & (simple_panel["date"] > pd.Timestamp("2024-02-15")))
    ]
    with pytest.warns(UserWarning, match="incomplete future window"):
        out = build_v1_snapshot(truncated, "2024-01-31")
    assert "B" not in out["account_id"].values
    assert "A" in out["account_id"].values


def test_v2_lead_screen_is_conditional_future_filter():
    dates = pd.date_range("2023-12-03", "2024-03-15", freq="D")
    rows = []
    for account_id, lead_drop in [("stable", False), ("lead_cross", True)]:
        for date in dates:
            balance = 1000.0
            if account_id == "lead_cross" and pd.Timestamp("2024-02-01") <= date <= pd.Timestamp("2024-02-14"):
                balance = 100.0
            if date >= pd.Timestamp("2024-02-15"):
                balance = 400.0
            rows.append(
                {
                    "date": date,
                    "account_id": account_id,
                    "account_type": "checking",
                    "product_group": "Checking_A",
                    "balance": balance,
                    "deposit_rate": 0.01,
                    "open_date": pd.Timestamp("2023-01-01"),
                    "status": "ACTIVE",
                    "is_internal": False,
                }
            )
    frame = pd.DataFrame(rows)
    out = build_v2_robustness_snapshot(frame, "2024-01-31", materiality_floor_units=100)
    assert set(out["account_id"]) == {"stable"}
