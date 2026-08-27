from __future__ import annotations

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

FEATURE_COLUMNS = [
    "asof_balance",
    "avg_bal_7d",
    "avg_bal_30d",
    "asof_vs_30d_avg_pct",
    "recent7_vs_30d_pct",
    "bal_std_30d",
    "bal_volatility_cv_30d",
    "min_bal_30d",
    "max_bal_30d",
    "bal_range_pct_30d",
    "n_obs_30d",
    "deposit_rate",
    "account_tenure_days",
    "account_type",
    "product_group",
    "deposit_rate_missing",
]


def prepare_model_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Add missingness indicator and align categorical dtypes."""
    out = frame.copy()
    out["deposit_rate_missing"] = out["deposit_rate"].isna().astype("int8")
    for col in ["account_type", "product_group"]:
        out[col] = out[col].astype("category")
    return out


def behavioral_baseline_score(frame: pd.DataFrame) -> pd.Series:
    """Higher score means stronger recent deterioration."""
    return -frame["recent7_vs_30d_pct"].astype(float)


def train_lightgbm(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    model_cfg: dict,
) -> lgb.LGBMClassifier:
    """Fit the same LightGBM family used in the original analysis."""
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=model_cfg.get("n_estimators", 1000),
        learning_rate=model_cfg.get("learning_rate", 0.05),
        num_leaves=model_cfg.get("num_leaves", 31),
        max_depth=-1,
        subsample=model_cfg.get("subsample", 0.8),
        subsample_freq=1,
        colsample_bytree=model_cfg.get("colsample_bytree", 0.8),
        reg_lambda=model_cfg.get("reg_lambda", 1.0),
        random_state=model_cfg.get("random_state", 42),
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        train[FEATURE_COLUMNS],
        train["runoff_flag"],
        eval_X=validation[FEATURE_COLUMNS],
        eval_y=validation["runoff_flag"],
        eval_metric="average_precision",
        categorical_feature=["account_type", "product_group"],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model


def evaluate_predictions(y_true: pd.Series, pred: pd.Series) -> dict[str, float]:
    """Return ranking metrics for a binary target."""
    return {
        "roc_auc": float(roc_auc_score(y_true, pred)),
        "pr_auc": float(average_precision_score(y_true, pred)),
        "prevalence": float(y_true.mean()),
    }
