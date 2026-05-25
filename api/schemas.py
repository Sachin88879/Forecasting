"""
Pydantic models used by the FastAPI request / response layer.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    """Body for the two `/forecast/*` endpoints."""

    insurance_type: str = Field(..., examples=["Private"])
    financial_class: str = Field(..., examples=["Inpatient"])
    horizon: int = Field(
        3,
        description="Number of future months to forecast (1, 3, or 6).",
        examples=[1, 3, 6],
    )


class InsurancePaidForecastItem(BaseModel):
    month: str
    insurance_paid: float
    insurance_paid_prophet: Optional[float] = None
    insurance_paid_xgboost: Optional[float] = None
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    sample_count: float


class DenialForecastItem(BaseModel):
    month: str
    denial_amount: float
    denial_amount_prophet: Optional[float] = None
    denial_amount_xgboost: Optional[float] = None
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    sample_count: float


class InsurancePaidForecastResponse(BaseModel):
    insurance_type: str
    financial_class: str
    horizon: int
    forecasts: List[InsurancePaidForecastItem]


class DenialForecastResponse(BaseModel):
    insurance_type: str
    financial_class: str
    horizon: int
    forecasts: List[DenialForecastItem]


class HealthResponse(BaseModel):
    status: str
    version: str
