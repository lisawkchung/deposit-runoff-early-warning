from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from deposit_runoff.config import load_config
from deposit_runoff.demo_data import make_demo_daily_panel
from deposit_runoff.features import build_v1_dataset
from deposit_runoff.metrics import top_fraction_metrics
from deposit_runoff.modeling import (
    behavioral_baseline_score,
    evaluate_predictions,
    prepare_model_frame,
    train_lightgbm,
)
from deposit_runoff.split import chronological_split
from deposit_runoff.targets import build_v2_robustness_dataset


def _all_snapshots(cfg: dict) -> list[str]:
    return [*map(str, cfg["snapshots"]["train"]), str(cfg["snapshots"]["validation"]), str(cfg["snapshots"]["test"])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/public.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    path = Path(cfg["data"]["path"])

    if path.exists():
        daily = pd.read_csv(path, parse_dates=["date", "open_date"])
    else:
        d = cfg["data"]
        daily = make_demo_daily_panel(
            n_accounts=d["n_accounts"], start_date=str(d["start_date"]), end_date=str(d["end_date"]), seed=cfg["seed"]
        )

    snapshots = _all_snapshots(cfg)
    v1 = build_v1_dataset(daily, snapshots, **cfg["v1"])
    v1 = prepare_model_frame(v1)
    train, val, test = chronological_split(
        v1,
        train_dates=cfg["snapshots"]["train"],
        validation_date=cfg["snapshots"]["validation"],
        test_date=cfg["snapshots"]["test"],
    )

    baseline = behavioral_baseline_score(val)
    print("validation baseline:", evaluate_predictions(val["runoff_flag"], baseline))

    model = train_lightgbm(train, val, cfg["model"])
    pred = model.predict_proba(test[model.feature_name_])[:, 1]
    print("test model:", evaluate_predictions(test["runoff_flag"], pred))
    print("test top 10%:", top_fraction_metrics(test, pred, fraction=0.10))

    v2 = build_v2_robustness_dataset(daily, snapshots, **cfg["v2_robustness"])
    print(
        "v2 robustness dataset:",
        {"rows": len(v2), "prevalence": float(v2["runoff_flag"].mean()) if len(v2) else None},
    )
    print("Synthetic metrics above are demo-only and are not the confidential portfolio results.")


if __name__ == "__main__":
    main()
