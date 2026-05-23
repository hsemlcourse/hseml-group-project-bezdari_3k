from src.ui.payload import build_feature_payload
from src.ui.streamlit_app import CLIENT_DEFAULTS


def test_build_feature_payload_converts_user_friendly_fields():
    payload = build_feature_payload(
        {
            "age_years": 42,
            "employment_years": 8,
            "income_total": 202500.0,
            "credit_amount": 406597.5,
            "annuity_amount": 24700.5,
            "goods_price": 351000.0,
            "ext_source_1": 0.35,
            "ext_source_2": 0.26,
            "ext_source_3": None,
            "gender": "M",
            "family_status": "Single / not married",
            "contract_type": "Cash loans",
            "education_type": "Higher education",
            "income_type": "Working",
            "occupation_type": "Core staff",
            "children_count": 1,
            "family_members": 2,
        }
    )

    assert payload["DAYS_BIRTH"] == -15330
    assert payload["DAYS_EMPLOYED"] == -2920
    assert payload["AMT_INCOME_TOTAL"] == 202500.0
    assert payload["AMT_CREDIT"] == 406597.5
    assert payload["AMT_ANNUITY"] == 24700.5
    assert payload["AMT_GOODS_PRICE"] == 351000.0
    assert payload["EXT_SOURCE_3"] is None
    assert payload["CODE_GENDER"] == "M"
    assert payload["CNT_CHILDREN"] == 1
    assert payload["CNT_FAM_MEMBERS"] == 2
    assert payload["NAME_EDUCATION_TYPE"] == "Higher education"
    assert payload["NAME_INCOME_TYPE"] == "Working"
    assert payload["OCCUPATION_TYPE"] == "Core staff"


def test_build_feature_payload_omits_empty_strings_and_keeps_zero_values():
    payload = build_feature_payload(
        {
            "age_years": 30,
            "employment_years": 0,
            "income_total": 0,
            "credit_amount": 100000.0,
            "annuity_amount": None,
            "goods_price": None,
            "gender": "",
            "family_status": "",
            "contract_type": "Revolving loans",
            "education_type": "",
            "income_type": "",
            "occupation_type": "",
            "children_count": 0,
            "family_members": 1,
        }
    )

    assert payload["DAYS_EMPLOYED"] == 0
    assert payload["AMT_INCOME_TOTAL"] == 0
    assert payload["CNT_CHILDREN"] == 0
    assert payload["CNT_FAM_MEMBERS"] == 1
    assert "CODE_GENDER" not in payload
    assert "NAME_FAMILY_STATUS" not in payload
    assert "NAME_EDUCATION_TYPE" not in payload
    assert payload["NAME_CONTRACT_TYPE"] == "Revolving loans"


def test_build_feature_payload_keeps_already_model_named_csv_fields():
    payload = build_feature_payload(
        {
            "SK_ID_CURR": 100001,
            "AMT_CREDIT": 500000.0,
            "AMT_INCOME_TOTAL": 150000.0,
            "EXT_SOURCE_2": 0.42,
        }
    )

    assert payload["SK_ID_CURR"] == 100001
    assert payload["AMT_CREDIT"] == 500000.0
    assert payload["AMT_INCOME_TOTAL"] == 150000.0
    assert payload["EXT_SOURCE_2"] == 0.42


def test_single_client_defaults_include_contract_type_used_by_model():
    payload = build_feature_payload(CLIENT_DEFAULTS)

    assert CLIENT_DEFAULTS["contract_type"] == "Cash loans"
    assert payload["NAME_CONTRACT_TYPE"] == "Cash loans"
