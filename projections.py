"""Wharton Laura Gao deterministic annual projection, beginning-of-year cash flows.

The model does NOT claim to size the 2033 operating reserve or a facility
contribution. They require separate liability and downside scenario analysis.
"""
from __future__ import annotations

import pandas as pd

FIRST_YEAR = 2027
LAST_YEAR = 2042
YEARLY_OPERATING_PAYMENT = 50_000.0


def project_wharton(
    annual_return: float,
    *,
    initial_contribution: float = 300_000.0,
    second_contribution: float = 150_000.0,
    facility_payment_2033: float = 0.0,
) -> pd.DataFrame:
    if not -1 < annual_return <= 2:
        raise ValueError("Annual return assumption must be between -100% and +200%.")
    if any(x < 0 for x in (initial_contribution, second_contribution, facility_payment_2033)):
        raise ValueError("Cash-flow amounts cannot be negative.")
    wealth = 0.0
    rows = []
    cumulative_shortfall = 0.0
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        previous = wealth
        contribution = (initial_contribution if year == 2027 else
                        second_contribution if year == 2028 else 0.0)
        required_operating = YEARLY_OPERATING_PAYMENT if 2033 <= year <= 2042 else 0.0
        requested_facility = facility_payment_2033 if year == 2033 else 0.0
        available = previous + contribution
        # Pay the operating obligation first; do not fund a facility at its expense.
        operating_paid = min(available, required_operating)
        available -= operating_paid
        facility_paid = min(available, requested_facility) if operating_paid >= required_operating else 0.0
        available -= facility_paid
        operating_shortfall = required_operating - operating_paid
        facility_shortfall = requested_facility - facility_paid
        cumulative_shortfall += operating_shortfall
        beginning_invested = available
        earnings = beginning_invested * annual_return
        wealth = max(0.0, beginning_invested + earnings)
        rows.append({
            "Year": year,
            "Opening balance": previous,
            "Beginning-year contribution": contribution,
            "Required operating payment": required_operating,
            "Operating payment funded": operating_paid,
            "Operating payment shortfall": operating_shortfall,
            "Facility payment requested": requested_facility,
            "Facility payment funded": facility_paid,
            "Facility payment shortfall": facility_shortfall,
            "Balance invested during year": beginning_invested,
            "Annual investment earnings": earnings,
            "End-of-year balance": wealth,
            "Cumulative operating shortfall": cumulative_shortfall,
        })
    return pd.DataFrame(rows)


def projection_issues(df: pd.DataFrame) -> list[str]:
    problems = []
    if df["Operating payment shortfall"].sum() > 0:
        problems.append("One or more required $50,000 operating payments could not be fully paid.")
    if df["Facility payment shortfall"].sum() > 0:
        problems.append("The entered facility withdrawal could not be fully paid.")
    return problems
