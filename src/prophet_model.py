"""
Train and use Prophet models, one per (insurance_type, financial_class) group,
for two targets: ``total_insurance_paid`` and ``total_denials``.

Run from the project root with:
    python -m src.prophet_model
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List

import pandas as pd

try:
    from prophet import Prophet
except ImportError:  # older releases
    from fbprophet import Prophet  # type: ignore

from config.holidays import get_prophet_holidays
from config.settings import (
    MONTHLY_DENIALS_FILE,
    MONTHLY_INSURANCE_PAID_FILE,
    PROPHET_DENIALS_FILE,
    PROPHET_INSURANCE_PAID_FILE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _group_key(insurance_type: str, financial_class: str) -> str:
    """Stable string key used in the saved-models dict."""
    return f"{insurance_type}__{financial_class}"


def _make_prophet() -> Prophet:
    """Return a Prophet instance configured with India-style holidays."""
    return Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,   # we forecast monthly
        daily_seasonality=False,
        holidays=get_prophet_holidays(),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _train_one_target(
    df: pd.DataFrame,
    value_col: str,
    save_path: Path,
) -> Dict[str, Prophet]:
    """
    Fit one Prophet model per (insurance_type, financial_class) group and
    pickle every model in a single dict for easy loading later.
    """
    models: Dict[str, Prophet] = {}

    groups = df[["insurance_type", "financial_class"]].drop_duplicates()
    for _, row in groups.iterrows():
        ins, fc = row["insurance_type"], row["financial_class"]
        sub = (
            df[(df["insurance_type"] == ins) & (df["financial_class"] == fc)]
            .sort_values("month")
            .rename(columns={"month": "ds", value_col: "y"})[["ds", "y"]]
        )
        if len(sub) < 3:        # not enough history for Prophet
            continue

        model = _make_prophet()
        model.fit(sub)
        models[_group_key(ins, fc)] = model

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("wb") as f:
        pickle.dump(models, f)
    print(f"Saved {len(models)} Prophet models -> {save_path.name}")
    return models


def train_all() -> None:
    """Train Prophet for both targets and save both bundles."""
    paid_df = pd.read_parquet(MONTHLY_INSURANCE_PAID_FILE)
    _train_one_target(paid_df, "total_insurance_paid", PROPHET_INSURANCE_PAID_FILE)

    den_df = pd.read_parquet(MONTHLY_DENIALS_FILE)
    _train_one_target(den_df, "total_denials", PROPHET_DENIALS_FILE)


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------
def _load_models(path: Path) -> Dict[str, Prophet]:
    with path.open("rb") as f:
        return pickle.load(f)


def forecast_group(
    models: Dict[str, Prophet],
    insurance_type: str,
    financial_class: str,
    horizon_months: int,
) -> List[dict]:
    """
    Forecast ``horizon_months`` future months for one group.
    Returns a list of dicts with ds / yhat / yhat_lower / yhat_upper.
    """
    key = _group_key(insurance_type, financial_class)
    if key not in models:
        return []

    model = models[key]
    future = model.make_future_dataframe(periods=horizon_months, freq="MS")
    fc = model.predict(future).tail(horizon_months)

    return [
        {
            "month": row["ds"].strftime("%Y-%m-%d"),
            "yhat": float(max(0.0, row["yhat"])),
            "yhat_lower": float(max(0.0, row["yhat_lower"])),
            "yhat_upper": float(max(0.0, row["yhat_upper"])),
        }
        for _, row in fc.iterrows()
    ]


def forecast_insurance_paid(
    insurance_type: str, financial_class: str, horizon_months: int
) -> List[dict]:
    return forecast_group(
        _load_models(PROPHET_INSURANCE_PAID_FILE),
        insurance_type,
        financial_class,
        horizon_months,
    )


def forecast_denials(
    insurance_type: str, financial_class: str, horizon_months: int
) -> List[dict]:
    return forecast_group(
        _load_models(PROPHET_DENIALS_FILE),
        insurance_type,
        financial_class,
        horizon_months,
    )


if __name__ == "__main__":
    train_all()
