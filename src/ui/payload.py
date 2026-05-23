"""Convert Streamlit form values into API feature payloads."""

from __future__ import annotations

from typing import Mapping


FIELD_MAPPING = {
    "income_total": "AMT_INCOME_TOTAL",
    "credit_amount": "AMT_CREDIT",
    "annuity_amount": "AMT_ANNUITY",
    "goods_price": "AMT_GOODS_PRICE",
    "ext_source_1": "EXT_SOURCE_1",
    "ext_source_2": "EXT_SOURCE_2",
    "ext_source_3": "EXT_SOURCE_3",
    "gender": "CODE_GENDER",
    "family_status": "NAME_FAMILY_STATUS",
    "contract_type": "NAME_CONTRACT_TYPE",
    "education_type": "NAME_EDUCATION_TYPE",
    "income_type": "NAME_INCOME_TYPE",
    "occupation_type": "OCCUPATION_TYPE",
    "children_count": "CNT_CHILDREN",
    "family_members": "CNT_FAM_MEMBERS",
}


def build_feature_payload(values: Mapping[str, object]) -> dict[str, object]:
    """Build a Home Credit feature payload from user-friendly form values."""

    payload: dict[str, object] = {}

    if values.get("age_years") is not None:
        payload["DAYS_BIRTH"] = -int(values["age_years"]) * 365
    if values.get("employment_years") is not None:
        payload["DAYS_EMPLOYED"] = -int(values["employment_years"]) * 365

    for source_name, target_name in FIELD_MAPPING.items():
        if source_name not in values:
            continue
        value = values[source_name]
        if value == "":
            continue
        payload[target_name] = value

    for key, value in values.items():
        if key.isupper() and value != "":
            payload[key] = value

    return payload
