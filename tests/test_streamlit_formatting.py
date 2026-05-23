from src.ui.formatting import format_probability, get_recommendation, risk_tone


def test_format_probability_uses_one_decimal_percent():
    assert format_probability(0.613) == "61.3%"


def test_risk_tone_returns_semantic_styles():
    assert risk_tone("low") == {"label": "Low Risk", "color": "#27ae60", "background": "#eaf8f0"}
    assert risk_tone("medium") == {"label": "Medium Risk", "color": "#b7791f", "background": "#fff8dd"}
    assert risk_tone("high") == {"label": "High Risk", "color": "#eb5757", "background": "#fdecec"}


def test_risk_tone_falls_back_to_neutral_style():
    assert risk_tone("unknown") == {"label": "Unknown Risk", "color": "#6b7280", "background": "#eef2f6"}


def test_get_recommendation_returns_analyst_facing_copy():
    assert get_recommendation("low") == "Acceptable risk profile"
    assert get_recommendation("medium") == "Manual review required"
    assert get_recommendation("high") == "High-risk application"
