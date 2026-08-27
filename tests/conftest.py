import pandas as pd
import pytest


@pytest.fixture
def simple_panel() -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", "2024-03-01", freq="D")
    rows = []
    for account_id, future_balance in [("A", 40.0), ("B", 80.0)]:
        for date in dates:
            balance = 100.0
            if date > pd.Timestamp("2024-01-31"):
                balance = future_balance
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
    return pd.DataFrame(rows)
