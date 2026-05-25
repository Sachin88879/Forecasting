"""
India-style holiday calendar.

Used in two places:
1. `feature_engineering.py` to flag `is_holiday`.
2. `prophet_model.py` as the `holidays` argument when fitting Prophet.
"""

from __future__ import annotations

import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    Holiday,
)


class IndiaHolidayCalendar(AbstractHolidayCalendar):
    """Minimal India-style holiday calendar (fixed-date holidays only)."""

    rules = [
        Holiday("New Year's Day", month=1, day=1),
        Holiday("Republic Day", month=1, day=26),
        Holiday("Labour Day", month=5, day=1),
        Holiday("Independence Day", month=8, day=15),
        Holiday("Gandhi Jayanti", month=10, day=2),
        Holiday("Christmas Day", month=12, day=25),
    ]


def get_holiday_dates(start: str = "2020-01-01", end: str = "2030-12-31") -> pd.DatetimeIndex:
    """Return a DatetimeIndex of all holidays between ``start`` and ``end``."""
    cal = IndiaHolidayCalendar()
    return cal.holidays(start=pd.Timestamp(start), end=pd.Timestamp(end))


def get_prophet_holidays(
    start: str = "2020-01-01", end: str = "2030-12-31"
) -> pd.DataFrame:
    """
    Return holidays as a DataFrame in the shape Prophet expects:
    columns ``holiday`` (name) and ``ds`` (date).
    """
    cal = IndiaHolidayCalendar()
    rows = []
    for rule in cal.rules:
        # `dates` returns every occurrence in the requested window
        for d in rule.dates(pd.Timestamp(start), pd.Timestamp(end)):
            rows.append({"holiday": rule.name, "ds": d})
    return pd.DataFrame(rows)
