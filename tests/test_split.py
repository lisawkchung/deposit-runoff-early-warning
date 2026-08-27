import pandas as pd
import pytest

from deposit_runoff.split import chronological_split


def test_chronological_split_is_forward_only():
    frame = pd.DataFrame(
        {
            "snapshot_date": pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"]),
            "runoff_flag": [0, 1, 0],
        }
    )
    train, val, test = chronological_split(
        frame,
        train_dates=["2024-01-31"],
        validation_date="2024-02-29",
        test_date="2024-03-31",
    )
    assert train["snapshot_date"].max() < val["snapshot_date"].min() < test["snapshot_date"].min()


def test_invalid_time_order_is_rejected():
    frame = pd.DataFrame(
        {"snapshot_date": pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"])}
    )
    with pytest.raises(ValueError, match="Chronological split"):
        chronological_split(
            frame,
            train_dates=["2024-03-31"],
            validation_date="2024-02-29",
            test_date="2024-01-31",
        )
