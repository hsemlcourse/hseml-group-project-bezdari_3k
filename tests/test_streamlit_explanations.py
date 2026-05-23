from src.ui.explanations import build_risk_drivers


def test_build_risk_drivers_flags_low_external_score_and_high_credit_ratio():
    drivers = build_risk_drivers(
        {
            "credit_amount": 900000.0,
            "income_total": 150000.0,
            "annuity_amount": 60000.0,
            "employment_years": 1,
            "ext_source_2": 0.18,
            "ext_source_3": 0.21,
            "family_status": "Single / not married",
        }
    )

    assert "Low EXT_SOURCE_2 increases risk" in drivers["increasing"]
    assert "High credit / income ratio increases risk" in drivers["increasing"]
    assert "Short employment history increases risk" in drivers["increasing"]


def test_build_risk_drivers_flags_stable_profile_as_reducing_risk():
    drivers = build_risk_drivers(
        {
            "credit_amount": 300000.0,
            "income_total": 220000.0,
            "annuity_amount": 18000.0,
            "employment_years": 9,
            "ext_source_2": 0.68,
            "ext_source_3": 0.72,
            "family_status": "Married",
        }
    )

    assert "Strong external score decreases risk" in drivers["reducing"]
    assert "Moderate annuity burden decreases risk" in drivers["reducing"]
    assert "Stable family status decreases risk" in drivers["reducing"]
