"""
Lightweight sanity tests for the inference layer.

These tests assume the full pipeline has been run:
    python -m src.preprocessing
    python -m src.feature_engineering
    python -m src.prophet_model
    python -m src.xgboost_model

Run with:
    pytest tests/ -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from config.settings import (
    MONTHLY_FEATURES_FILE,
    PROPHET_INSURANCE_PAID_FILE,
    XGB_INSURANCE_PAID_FILE,
)
from src import inference


HAS_DATA = MONTHLY_FEATURES_FILE.exists()
HAS_MODELS = PROPHET_INSURANCE_PAID_FILE.exists() and XGB_INSURANCE_PAID_FILE.exists()

skip_if_missing = pytest.mark.skipif(
    not (HAS_DATA and HAS_MODELS),
    reason="Run preprocessing + training before executing these tests.",
)


def _first_group() -> tuple[str, str]:
    import pandas as pd

    df = pd.read_parquet(MONTHLY_FEATURES_FILE)
    row = df.iloc[0]
    return row["insurance_type"], row["financial_class"]


def test_validate_horizon_rejects_invalid_value():
    with pytest.raises(ValueError):
        inference.forecast_insurance_paid("Private", "Inpatient", horizon=99)


@skip_if_missing
def test_insurance_paid_forecast_returns_horizon_rows():
    ins, fc = _first_group()
    out = inference.forecast_insurance_paid(ins, fc, horizon=3)
    assert len(out) == 3
    for item in out:
        assert item["month"]
        assert item["insurance_paid"] >= 0
        assert "sample_count" in item


@skip_if_missing
def test_denials_forecast_returns_horizon_rows():
    ins, fc = _first_group()
    out = inference.forecast_denials(ins, fc, horizon=1)
    assert len(out) == 1
    assert out[0]["denial_amount"] >= 0
