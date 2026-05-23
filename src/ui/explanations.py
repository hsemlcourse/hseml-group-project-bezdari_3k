"""Transparent local risk-driver rules for the Streamlit demo."""

from __future__ import annotations

from typing import Mapping


def _float_value(values: Mapping[str, object], key: str, default: float = 0.0) -> float:
    value = values.get(key, default)
    if value is None or value == "":
        return default
    return float(value)


def build_risk_drivers(values: Mapping[str, object]) -> dict[str, list[str]]:
    """Build concise risk-driver text from visible form fields.

    These are deterministic UI explanations for the demo. They do not claim to be SHAP values.
    """

    income = max(_float_value(values, "income_total"), 1.0)
    credit = _float_value(values, "credit_amount")
    annuity = _float_value(values, "annuity_amount")
    employment_years = _float_value(values, "employment_years")
    ext_source_2 = _float_value(values, "ext_source_2", 0.5)
    ext_source_3 = _float_value(values, "ext_source_3", 0.5)
    family_status = str(values.get("family_status", ""))

    credit_income_ratio = credit / income
    annuity_income_ratio = annuity / income

    increasing: list[str] = []
    reducing: list[str] = []

    if ext_source_2 < 0.3:
        increasing.append("Low EXT_SOURCE_2 increases risk")
    if ext_source_3 < 0.3:
        increasing.append("Low EXT_SOURCE_3 increases risk")
    if credit_income_ratio > 4:
        increasing.append("High credit / income ratio increases risk")
    if annuity_income_ratio > 0.25:
        increasing.append("High annuity / income ratio increases risk")
    if employment_years < 2:
        increasing.append("Short employment history increases risk")

    if ext_source_2 >= 0.6 or ext_source_3 >= 0.6:
        reducing.append("Strong external score decreases risk")
    if annuity_income_ratio <= 0.12:
        reducing.append("Moderate annuity burden decreases risk")
    if employment_years >= 5:
        reducing.append("Stable employment decreases risk")
    if family_status in {"Married", "Civil marriage"}:
        reducing.append("Stable family status decreases risk")

    if not increasing:
        increasing.append("No strong risk-increasing rule triggered")
    if not reducing:
        reducing.append("No strong risk-reducing rule triggered")

    return {"increasing": increasing, "reducing": reducing}
