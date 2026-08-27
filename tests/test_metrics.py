import pandas as pd

from deposit_runoff.metrics import balance_extreme_capture, top_fraction_metrics


def test_top_fraction_metrics_uses_only_labeled_severity():
    frame = pd.DataFrame(
        {
            "runoff_flag": [1, 0, 1, 0],
            "runoff_amount": [100.0, 9999.0, 50.0, 9999.0],
        }
    )
    result = top_fraction_metrics(frame, [4, 3, 2, 1], fraction=0.50)
    assert result["event_capture"] == 0.5
    assert round(result["severity_capture"], 6) == round(100 / 150, 6)


def test_balance_ranking_can_optimize_different_objective():
    frame = pd.DataFrame(
        {
            "avg_bal_30d": [10.0, 20.0, 1000.0, 2000.0],
            "runoff_flag": [1, 1, 1, 0],
            "runoff_amount": [5.0, 10.0, 800.0, 0.0],
        }
    )
    large = balance_extreme_capture(frame, largest_first=True, fraction=0.50)
    small = balance_extreme_capture(frame, largest_first=False, fraction=0.50)
    assert large["severity_capture"] > small["severity_capture"]
    assert small["event_capture"] > large["event_capture"]
