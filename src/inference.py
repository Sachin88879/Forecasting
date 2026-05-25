"""
Glue layer used by the FastAPI app.

It calls both Prophet and XGBoost for a given (insurance_type, financial_class)
group, merges the per-month outputs, and adds a `sample_count` forecast.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from config.settings import (
    ALLOWED_HORIZONS,
    MONTHLY_FEATURES_FILE,
)
from src import prophet_model, xgboost_model


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _validate_horizon(horizon: int) -> None:
    if horizon not in ALLOWED_HORIZONS:
        raise ValueError(
            f"horizon must be one of {ALLOWED_HORIZONS}, got {horizon}"
        )


# ---------------------------------------------------------------------------
# Sample-count forecast (simple seasonal-naive baseline)
# ---------------------------------------------------------------------------
def _forecast_sample_count(
    insurance_type: str, financial_class: str, horizon: int
) -> Dict[str, float]:
    """Forecast sample_count using last-3-month rolling average per group."""
    try:
        df = pd.read_parquet(MONTHLY_FEATURES_FILE)
    except FileNotFoundError:
        return {}

    sub = df[
        (df["insurance_type"] == insurance_type)
        & (df["financial_class"] == financial_class)
    ].sort_values("month")
    if sub.empty:
        return {}

    baseline = float(sub["sample_count"].tail(3).mean())
    last_month = pd.to_datetime(sub["month"].max())
    out: Dict[str, float] = {}
    for i in range(horizon):
        fm = (last_month + pd.DateOffset(months=i + 1)).strftime("%Y-%m-%d")
        out[fm] = round(baseline, 2)
    return out


# ---------------------------------------------------------------------------
# Public entry points used by the API
# ---------------------------------------------------------------------------
def _ensemble_value(prophet_val: float | None, xgb_val: float | None) -> float:
    """Simple average of the two model outputs when both exist."""
    vals = [v for v in [prophet_val, xgb_val] if v is not None]
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def forecast_insurance_paid(
    insurance_type: str, financial_class: str, horizon: int
) -> List[dict]:
    """Forecast monthly insurance paid for the next ``horizon`` months."""
    _validate_horizon(horizon)

    prophet_fc = prophet_model.forecast_insurance_paid(
        insurance_type, financial_class, horizon
    )
    xgb_fc = xgboost_model.forecast_insurance_paid(
        insurance_type, financial_class, horizon
    )
    sample_fc = _forecast_sample_count(insurance_type, financial_class, horizon)

    prophet_by_month = {p["month"]: p for p in prophet_fc}
    xgb_by_month = {x["month"]: x for x in xgb_fc}
    months = sorted(set(prophet_by_month) | set(xgb_by_month))

    results: List[dict] = []
    for m in months:
        p = prophet_by_month.get(m)
        x = xgb_by_month.get(m)
        results.append(
            {
                "month": m,
                "insurance_paid": _ensemble_value(
                    p["yhat"] if p else None,
                    x["yhat"] if x else None,
                ),
                "insurance_paid_prophet": p["yhat"] if p else None,
                "insurance_paid_xgboost": x["yhat"] if x else None,
                "lower_bound": p["yhat_lower"] if p else None,
                "upper_bound": p["yhat_upper"] if p else None,
                "sample_count": sample_fc.get(m, 0.0),
            }
        )
    return results


def forecast_denials(
    insurance_type: str, financial_class: str, horizon: int
) -> List[dict]:
    """Forecast monthly denial amounts for the next ``horizon`` months."""
    _validate_horizon(horizon)

    prophet_fc = prophet_model.forecast_denials(
        insurance_type, financial_class, horizon
    )
    xgb_fc = xgboost_model.forecast_denials(
        insurance_type, financial_class, horizon
    )
    sample_fc = _forecast_sample_count(insurance_type, financial_class, horizon)

    prophet_by_month = {p["month"]: p for p in prophet_fc}
    xgb_by_month = {x["month"]: x for x in xgb_fc}
    months = sorted(set(prophet_by_month) | set(xgb_by_month))

    results: List[dict] = []
    for m in months:
        p = prophet_by_month.get(m)
        x = xgb_by_month.get(m)
        ensemble = _ensemble_value(
            p["yhat"] if p else None,
            x["yhat"] if x else None,
        )
        # Rough denial-count estimate: denial amount / average-per-claim (or fall back to sample_count)
        results.append(
            {
                "month": m,
                "denial_amount": ensemble,
                "denial_amount_prophet": p["yhat"] if p else None,
                "denial_amount_xgboost": x["yhat"] if x else None,
                "lower_bound": p["yhat_lower"] if p else None,
                "upper_bound": p["yhat_upper"] if p else None,
                "sample_count": sample_fc.get(m, 0.0),
            }
        )
    return results
