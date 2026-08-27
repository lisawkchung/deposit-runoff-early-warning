# Data

The institutional dataset used for the original analysis is confidential and is **not** included in this repository.

## Generic input schema

`run_analysis.py` expects one row per account per calendar date with these columns:

| Column | Type | Meaning |
|---|---|---|
| `date` | date | Observation date |
| `account_id` | string | Stable de-identified account key |
| `account_type` | string | Generic category such as `checking` or `savings` |
| `product_group` | string | Sanitized product grouping |
| `balance` | float | End-of-day balance |
| `deposit_rate` | float / null | Account deposit rate |
| `open_date` | date | Account open date |
| `status` | string | Account status; public pipeline expects `ACTIVE` for live relationships |
| `is_internal` | bool | Whether the row belongs to an internal/non-customer product |

## Public demo

Generate deterministic synthetic data with:

```bash
python scripts/make_demo_data.py --config config/public.yaml
```

The synthetic data are for code-path reproduction only. Their metrics are **not** intended to reproduce the confidential portfolio's reported results.
