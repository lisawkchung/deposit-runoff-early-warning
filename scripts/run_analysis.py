from __future__ import annotations

import argparse

import pandas as pd

from deposit_runoff.config import load_config
from deposit_runoff.features import build_v1_dataset
from deposit_runoff.modeling import prepare_model_frame, train_lightgbm
from deposit_runoff.split import chronological_split
from deposit_runoff.targets import build_v2_robustness_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    daily = pd.read_csv(cfg["data"]["path"], parse_dates=["date", "open_date"])
    snapshots = [
        *map(str, cfg["snapshots"]["train"]),
        str(cfg["snapshots"]["validation"]),
        str(cfg["snapshots"]["test"]),
    ]

    v1 = prepare_model_frame(build_v1_dataset(daily, snapshots, **cfg["v1"]))
    train, val, test = chronological_split(
        v1,
        train_dates=cfg["snapshots"]["train"],
        validation_date=cfg["snapshots"]["validation"],
        test_date=cfg["snapshots"]["test"],
    )
    model = train_lightgbm(train, val, cfg["model"])
    test_pred = model.predict_proba(test[model.feature_name_])[:, 1]
    test = test.assign(pred_risk=test_pred)

    v2 = build_v2_robustness_dataset(daily, snapshots, **cfg["v2_robustness"])
    print(f"V1 snapshots: {len(v1):,}; V2 robustness rows: {len(v2):,}")
    print("Analysis completed. Keep authorized data and outputs outside the public repository.")


if __name__ == "__main__":
    main()
