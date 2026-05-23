"""Presentation helpers for the Streamlit risk console."""

from __future__ import annotations


RISK_TONES = {
    "low": {"label": "Low Risk", "color": "#27ae60", "background": "#eaf8f0"},
    "medium": {"label": "Medium Risk", "color": "#b7791f", "background": "#fff8dd"},
    "high": {"label": "High Risk", "color": "#eb5757", "background": "#fdecec"},
}
UNKNOWN_RISK_TONE = {"label": "Unknown Risk", "color": "#6b7280", "background": "#eef2f6"}

RECOMMENDATIONS = {
    "low": "Acceptable risk profile",
    "medium": "Manual review required",
    "high": "High-risk application",
}


def format_probability(probability: float) -> str:
    """Format a probability as a one-decimal percentage."""

    return f"{probability * 100:.1f}%"


def risk_tone(risk_level: str) -> dict[str, str]:
    """Return display label and colors for a risk level."""

    return RISK_TONES.get(risk_level.lower(), UNKNOWN_RISK_TONE)


def get_recommendation(risk_level: str) -> str:
    """Return analyst-facing recommendation text for a risk level."""

    return RECOMMENDATIONS.get(risk_level.lower(), "Review required")
