from __future__ import annotations

import argparse
from pathlib import Path

from deposit_runoff.config import load_config
from deposit_runoff.demo_data import make_demo_daily_panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/public.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]

    frame = make_demo_daily_panel(
        n_accounts=data_cfg["n_accounts"],
        start_date=str(data_cfg["start_date"]),
        end_date=str(data_cfg["end_date"]),
        seed=cfg["seed"],
    )
    path = Path(data_cfg["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"wrote {len(frame):,} synthetic account-day rows to {path}")


if __name__ == "__main__":
    main()
