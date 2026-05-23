"""Streamlit fintech dashboard for the CP3 Home Credit deploy demo."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st

from src.ui.explanations import build_risk_drivers
from src.ui.formatting import format_probability, risk_tone
from src.ui.i18n import (
    LANGUAGE_LABELS,
    language_code_from_label,
    localized_recommendation,
    localize_risk_tone,
    page_options,
    text,
    translate_driver,
    translate_option,
)
from src.ui.payload import build_feature_payload


DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
ROOT_DIR = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT_DIR / "report"
IMAGE_DIR = REPORT_DIR / "images"
EXPERIMENTS_PATH = ROOT_DIR / "models" / "experiment_results.csv"
FEATURES_PATH = REPORT_DIR / "data_quality" / "selected_features.csv"

CLIENT_DEFAULTS = {
    "age_years": 35,
    "gender": "M",
    "contract_type": "Cash loans",
    "education_type": "Secondary / secondary special",
    "family_status": "Single / not married",
    "income_type": "Working",
    "occupation_type": "Core staff",
    "children_count": 0,
    "family_members": 2,
    "income_total": 150000.0,
    "credit_amount": 500000.0,
    "annuity_amount": 25000.0,
    "goods_price": 450000.0,
    "ext_source_1": 0.50,
    "ext_source_2": 0.50,
    "ext_source_3": 0.50,
    "employment_years": 5,
}

CLIENT_STRESSED_PROFILE = {
    **CLIENT_DEFAULTS,
    "age_years": 20,
    "income_total": 30000.0,
    "credit_amount": 1000000.0,
    "annuity_amount": 100000.0,
    "goods_price": 1000000.0,
    "ext_source_1": 0.0,
    "ext_source_2": 0.0,
    "ext_source_3": 0.0,
    "employment_years": 0,
}

CLIENT_STATE_PREFIX = "client_"


def inject_css() -> None:
    """Apply the fintech dashboard visual system."""

    st.markdown(
        """
        <style>
        :root {
            --bg: #f7f8fa;
            --surface: #ffffff;
            --text: #1f2937;
            --muted: #6b7280;
            --border: #e5e7eb;
            --primary: #2f80ed;
            --primary-dark: #1f67c7;
            --low: #27ae60;
            --medium: #f2c94c;
            --high: #eb5757;
            --shadow: 0 8px 24px rgba(31, 41, 55, 0.06);
        }

        .stApp {
            background: var(--bg);
            color: var(--text);
        }

        [data-testid="stHeader"] {
            height: 56px;
            background: rgba(247, 248, 250, 0.94);
            border-bottom: 1px solid rgba(229, 231, 235, 0.75);
            backdrop-filter: blur(12px);
        }

        [data-testid="stSidebar"] {
            background: var(--surface);
            border-right: 1px solid var(--border);
        }

        .block-container {
            max-width: 1280px;
            padding-top: 5.2rem;
            padding-bottom: 2.4rem;
        }

        h1, h2, h3, h4 {
            letter-spacing: 0;
            color: var(--text);
        }

        .page-header {
            position: relative;
            max-width: 980px;
            margin: 0 0 1.35rem;
            padding-left: 1rem;
            border-left: 4px solid var(--primary);
        }

        .dashboard-title {
            display: block;
            overflow: visible;
            white-space: normal;
            word-break: normal;
            font-size: 2.05rem;
            line-height: 1.18;
            font-weight: 790;
            margin: 0 0 0.35rem;
            padding: 0;
        }

        .dashboard-subtitle {
            color: var(--muted);
            font-size: 0.98rem;
            line-height: 1.55;
            margin: 0;
        }

        .brand-row {
            display: flex;
            align-items: center;
            gap: 0.7rem;
            margin-bottom: 0.75rem;
        }

        .brand-mark {
            width: 42px;
            height: 42px;
            border-radius: 8px;
            display: inline-grid;
            place-items: center;
            background: var(--primary);
            color: #ffffff;
            font-weight: 800;
            box-shadow: var(--shadow);
        }

        .brand-title {
            font-size: 1.03rem;
            font-weight: 760;
            line-height: 1.18;
        }

        .brand-caption,
        .muted {
            color: var(--muted);
            font-size: 0.86rem;
        }

        [data-testid="stSidebar"] [role="radiogroup"] {
            display: flex;
            flex-direction: column;
            gap: 0.2rem;
            margin-top: 0.35rem;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label[data-baseweb="radio"] {
            display: flex;
            align-items: center;
            width: 100%;
            border: 1px solid transparent;
            border-radius: 8px;
            padding: 0.58rem 0.72rem;
            margin: 0;
            min-height: 2.35rem;
            color: var(--text);
            transition: background 120ms ease, border-color 120ms ease, color 120ms ease;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label[data-baseweb="radio"]:hover {
            background: #f3f7fd;
            border-color: #d8e6f8;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) {
            background: #edf5ff;
            border-color: #b9d9ff;
            color: var(--primary-dark);
            font-weight: 760;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label[data-baseweb="radio"] > div:first-child {
            display: none;
        }

        [data-testid="stSidebar"] [role="radiogroup"] input[type="radio"] {
            opacity: 0;
            width: 0;
            min-width: 0;
            margin: 0;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label[data-baseweb="radio"] > div:last-child {
            margin-left: 0;
        }

        [data-testid="stSidebar"] [role="radiogroup"] p {
            font-size: 0.92rem;
            line-height: 1.25;
            margin: 0;
        }

        [data-testid="stSidebar"] input {
            border-radius: 8px;
        }

        [data-testid="stMetric"] {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: var(--shadow);
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            box-shadow: var(--shadow);
        }

        div[data-testid="stForm"] {
            border: 0;
            padding: 0;
        }

        .stButton > button,
        .stFormSubmitButton > button,
        [data-testid="stBaseButton-primary"],
        .stDownloadButton > button {
            border-radius: 8px !important;
            border: 1px solid var(--primary-dark) !important;
            background: var(--primary) !important;
            color: #ffffff !important;
            font-weight: 700 !important;
            min-height: 2.7rem;
        }

        .stButton > button:hover,
        .stFormSubmitButton > button:hover,
        .stDownloadButton > button:hover {
            background: var(--primary-dark) !important;
            border-color: var(--primary-dark) !important;
        }

        .status-chip,
        .risk-badge {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.35rem 0.78rem;
            font-weight: 760;
            font-size: 0.84rem;
            border: 1px solid transparent;
        }

        .status-ok {
            color: #1b7f46;
            background: #eaf8f0;
            border-color: #c6efd6;
        }

        .status-warn {
            color: #9a6700;
            background: #fff8dd;
            border-color: #f7e5a3;
        }

        .status-error {
            color: #b42318;
            background: #fdecec;
            border-color: #fac5c5;
        }

        .result-panel {
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.25rem;
            background: var(--surface);
            box-shadow: var(--shadow);
        }

        .result-label {
            color: var(--muted);
            font-size: 0.82rem;
            font-weight: 720;
            text-transform: uppercase;
            letter-spacing: 0;
        }

        .probability {
            font-size: 3rem;
            font-weight: 820;
            line-height: 1;
            margin: 0.45rem 0 0.85rem;
        }

        .meter {
            height: 12px;
            background: #edf1f5;
            border-radius: 999px;
            overflow: hidden;
            border: 1px solid var(--border);
            margin-top: 0.85rem;
        }

        .meter-fill {
            height: 100%;
            border-radius: 999px;
        }

        .driver-box {
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
            background: var(--surface);
        }

        .driver-box strong {
            display: block;
            margin-bottom: 0.5rem;
        }

        .callout {
            border: 1px solid var(--border);
            border-left: 4px solid var(--primary);
            border-radius: 8px;
            padding: 1rem;
            background: var(--surface);
            color: var(--text);
        }

        .small-table-note {
            color: var(--muted);
            font-size: 0.82rem;
        }

        @media (max-width: 900px) {
            .block-container {
                padding-top: 4.6rem;
            }

            .dashboard-title {
                font-size: 1.7rem;
                line-height: 1.22;
            }

            .probability {
                font-size: 2.35rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def normalize_api_url(api_base_url: str) -> str:
    """Normalize API URL from sidebar/env."""

    return api_base_url.strip().rstrip("/")


def get_json(url: str, timeout: float = 2.0) -> tuple[dict[str, Any] | None, str | None]:
    """GET JSON with concise error reporting."""

    try:
        response = requests.get(url, timeout=timeout)
    except requests.RequestException as exc:
        return None, f"Connection error: {exc}"
    try:
        data = response.json()
    except ValueError:
        data = {"detail": response.text}
    if response.status_code >= 400:
        return None, str(data.get("detail", response.text))
    return data, None


def post_json(url: str, payload: dict[str, Any], timeout: float = 8.0) -> tuple[dict[str, Any] | None, str | None]:
    """POST JSON with concise error reporting."""

    try:
        response = requests.post(url, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        return None, f"Connection error: {exc}"
    try:
        data = response.json()
    except ValueError:
        data = {"detail": response.text}
    if response.status_code >= 400:
        return None, str(data.get("detail", response.text))
    return data, None


@st.cache_data(show_spinner=False)
def load_experiment_results() -> pd.DataFrame:
    """Load committed experiment table if available."""

    if not EXPERIMENTS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(EXPERIMENTS_PATH)


@st.cache_data(show_spinner=False)
def load_selected_features() -> pd.DataFrame:
    """Load selected feature importance table if available."""

    if not FEATURES_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(FEATURES_PATH)


def best_model_row() -> pd.Series | None:
    """Return best committed experiment row."""

    results = load_experiment_results()
    if results.empty or "test_roc_auc" not in results.columns:
        return None
    return results.sort_values("test_roc_auc", ascending=False).iloc[0]


@st.cache_data(show_spinner=False)
def load_missing_values_preview() -> pd.DataFrame:
    """Load missing-values preview table for the data explorer page."""

    missing_path = REPORT_DIR / "data_quality" / "missing_values.csv"
    if not missing_path.exists():
        return pd.DataFrame()
    return pd.read_csv(missing_path).head(20)


def render_sidebar(default_api_base_url: str) -> tuple[str, str, dict[str, Any] | None, str]:
    """Render sidebar navigation and model status."""

    with st.sidebar:
        language_label = st.selectbox("Language / Язык", list(LANGUAGE_LABELS.values()), key="ui_language_label")
        lang = language_code_from_label(language_label)
        nav_options = page_options(lang)
        page_labels_by_key = {page_key: label for label, page_key in nav_options.items()}

        st.markdown(
            f"""
            <div class="brand-row">
                <div class="brand-mark">HC</div>
                <div>
                    <div class="brand-title">Home Credit<br>Risk Scoring</div>
                    <div class="brand-caption">{text(lang, "sidebar.brand_caption")}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        page = st.radio(
            text(lang, "sidebar.navigation"),
            list(page_labels_by_key),
            format_func=page_labels_by_key.__getitem__,
            key="selected_page_key",
            label_visibility="collapsed",
        )
        st.divider()
        api_base_url = normalize_api_url(st.text_input(text(lang, "sidebar.api_base_url"), value=default_api_base_url))
        health, health_error = get_json(f"{api_base_url}/health")

        st.markdown(f"#### {text(lang, 'sidebar.api_health')}")
        if health_error:
            st.markdown(
                f'<span class="status-chip status-error">{text(lang, "sidebar.api_unavailable")}</span>',
                unsafe_allow_html=True,
            )
            st.caption(health_error)
        elif health and health.get("model_loaded"):
            st.markdown(
                f'<span class="status-chip status-ok">{text(lang, "sidebar.model_loaded")}</span>',
                unsafe_allow_html=True,
            )
            st.caption(str(health.get("detail", "")))
        else:
            st.markdown(
                f'<span class="status-chip status-warn">{text(lang, "sidebar.model_missing")}</span>',
                unsafe_allow_html=True,
            )
            st.caption(str((health or {}).get("detail", text(lang, "sidebar.model_missing_detail"))))

        st.link_button(text(lang, "sidebar.open_swagger"), f"{api_base_url}/docs", width="stretch")
        return page, api_base_url, health, lang


def render_header(title: str, subtitle: str) -> None:
    """Render consistent page header."""

    st.markdown(
        f"""
        <div class="page-header">
            <div class="dashboard-title">{title}</div>
            <div class="dashboard-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview(lang: str) -> None:
    """Overview dashboard page."""

    render_header(text(lang, "page.overview.title"), text(lang, "page.overview.subtitle"))
    best = best_model_row()
    auc_value = f"{float(best['test_roc_auc']):.3f}" if best is not None else "n/a"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(text(lang, "overview.total_applications"), "307,511")
    col2.metric(text(lang, "overview.default_rate"), "8.1%")
    col3.metric(text(lang, "overview.model_auc"), auc_value)
    col4.metric(text(lang, "overview.threshold"), "0.35")

    left, right = st.columns([1.25, 0.75], gap="large")
    with left:
        with st.container(border=True):
            st.markdown(f"### {text(lang, 'overview.project_goal')}")
            st.write(text(lang, "overview.project_goal_text"))
            st.markdown(
                f'<span class="status-chip status-warn">{text(lang, "overview.recommendation_review")}</span>',
                unsafe_allow_html=True,
            )
    with right:
        with st.container(border=True):
            st.markdown(f"### {text(lang, 'overview.data_scope')}")
            st.write(text(lang, "overview.data_scope_text"))
            st.caption(text(lang, "overview.target_note"))


def client_state_key(name: str) -> str:
    """Return stable Streamlit key for a single-client input."""

    return f"{CLIENT_STATE_PREFIX}{name}"


def ensure_client_state_defaults() -> None:
    """Initialize single-client inputs once per Streamlit session."""

    for key, value in CLIENT_DEFAULTS.items():
        st.session_state.setdefault(client_state_key(key), value)


def apply_client_profile(values: dict[str, object]) -> None:
    """Apply a demo profile to visible single-client controls."""

    for key, value in values.items():
        st.session_state[client_state_key(key)] = value


def get_client_input_values() -> dict[str, object]:
    """Read current single-client values from Streamlit state."""

    return {key: st.session_state[client_state_key(key)] for key in CLIENT_DEFAULTS}


def collect_single_client_values(lang: str) -> tuple[bool, dict[str, object]]:
    """Render single-client scoring form."""

    ensure_client_state_defaults()

    preset_cols = st.columns([1, 1, 2.5])
    with preset_cols[0]:
        if st.button(text(lang, "single.standard_profile"), width="stretch"):
            apply_client_profile(CLIENT_DEFAULTS)
    with preset_cols[1]:
        if st.button(text(lang, "single.stressed_profile"), width="stretch"):
            apply_client_profile(CLIENT_STRESSED_PROFILE)

    st.markdown(f"#### {text(lang, 'single.client_profile')}")
    profile_cols = st.columns(4)
    with profile_cols[0]:
        st.slider(text(lang, "field.age"), 18, 75, key=client_state_key("age_years"))
        st.selectbox(
            text(lang, "field.gender"),
            ["M", "F", "XNA"],
            key=client_state_key("gender"),
            format_func=lambda option: translate_option(lang, option),
        )
    with profile_cols[1]:
        st.selectbox(
            text(lang, "field.education"),
            [
                "Secondary / secondary special",
                "Higher education",
                "Incomplete higher",
                "Lower secondary",
                "Academic degree",
            ],
            key=client_state_key("education_type"),
            format_func=lambda option: translate_option(lang, option),
        )
        st.selectbox(
            text(lang, "field.family_status"),
            ["Single / not married", "Married", "Civil marriage", "Separated", "Widow"],
            key=client_state_key("family_status"),
            format_func=lambda option: translate_option(lang, option),
        )
    with profile_cols[2]:
        st.selectbox(
            text(lang, "field.income_type"),
            ["Working", "Commercial associate", "Pensioner", "State servant", "Student"],
            key=client_state_key("income_type"),
            format_func=lambda option: translate_option(lang, option),
        )
        st.selectbox(
            text(lang, "field.occupation_type"),
            ["Laborers", "Core staff", "Managers", "Sales staff", "Drivers", "Accountants", "Other"],
            key=client_state_key("occupation_type"),
            format_func=lambda option: translate_option(lang, option),
        )
    with profile_cols[3]:
        st.slider(text(lang, "field.children_count"), 0, 10, key=client_state_key("children_count"))
        st.slider(text(lang, "field.family_members"), 1, 10, key=client_state_key("family_members"))

    st.markdown(f"#### {text(lang, 'single.loan_parameters')}")
    contract_cols = st.columns([1, 3])
    with contract_cols[0]:
        st.selectbox(
            text(lang, "field.contract_type"),
            ["Cash loans", "Revolving loans"],
            key=client_state_key("contract_type"),
            format_func=lambda option: translate_option(lang, option),
        )

    loan_cols = st.columns(4)
    with loan_cols[0]:
        st.number_input(
            text(lang, "field.total_income"),
            min_value=0.0,
            step=10000.0,
            key=client_state_key("income_total"),
        )
    with loan_cols[1]:
        st.number_input(
            text(lang, "field.credit_amount"),
            min_value=0.0,
            step=10000.0,
            key=client_state_key("credit_amount"),
        )
    with loan_cols[2]:
        st.number_input(text(lang, "field.annuity"), min_value=0.0, step=1000.0, key=client_state_key("annuity_amount"))
    with loan_cols[3]:
        st.number_input(
            text(lang, "field.goods_price"),
            min_value=0.0,
            step=10000.0,
            key=client_state_key("goods_price"),
        )

    st.markdown(f"#### {text(lang, 'single.external_history')}")
    history_cols = st.columns(4)
    with history_cols[0]:
        st.slider("EXT_SOURCE_1", 0.0, 1.0, step=0.01, key=client_state_key("ext_source_1"))
    with history_cols[1]:
        st.slider("EXT_SOURCE_2", 0.0, 1.0, step=0.01, key=client_state_key("ext_source_2"))
    with history_cols[2]:
        st.slider("EXT_SOURCE_3", 0.0, 1.0, step=0.01, key=client_state_key("ext_source_3"))
    with history_cols[3]:
        st.slider(text(lang, "field.employment_years"), 0, 45, key=client_state_key("employment_years"))

    submitted = st.button(text(lang, "single.calculate"), type="primary", width="stretch")
    return submitted, get_client_input_values()


def render_prediction_result(
    prediction: dict[str, Any] | None,
    error: str | None,
    values: dict[str, object],
    lang: str,
) -> None:
    """Render result block for one applicant."""

    if error:
        st.error(error)
        return
    if not prediction:
        st.markdown(
            f"""
            <div class="result-panel">
                <div class="result-label">{text(lang, "result.probability")}</div>
                <div class="probability">{text(lang, "result.waiting")}</div>
                <div class="muted">{text(lang, "result.waiting_hint")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    probability = float(prediction["default_probability"])
    risk_level = str(prediction["risk_level"])
    tone = localize_risk_tone(lang, risk_level, risk_tone(risk_level))
    recommendation = localized_recommendation(lang, risk_level)
    width = min(max(probability * 100, 0), 100)

    st.markdown(
        f"""
        <div class="result-panel">
            <div class="result-label">{text(lang, "result.probability")}</div>
            <div class="probability">{format_probability(probability)}</div>
            <div class="risk-badge" style="color:{tone['color']}; background:{tone['background']}">
                {tone['label']}
            </div>
            <div class="meter">
                <div class="meter-fill" style="width:{width:.1f}%; background:{tone['color']}"></div>
            </div>
            <p><strong>{text(lang, "result.recommendation")}:</strong> {recommendation}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    drivers = build_risk_drivers(values)
    inc_col, dec_col = st.columns(2)
    with inc_col:
        st.markdown(
            f'<div class="driver-box"><strong>{text(lang, "result.increasing")}</strong>',
            unsafe_allow_html=True,
        )
        for item in drivers["increasing"]:
            st.write(f"+ {translate_driver(lang, item)}")
        st.markdown("</div>", unsafe_allow_html=True)
    with dec_col:
        st.markdown(
            f'<div class="driver-box"><strong>{text(lang, "result.reducing")}</strong>',
            unsafe_allow_html=True,
        )
        for item in drivers["reducing"]:
            st.write(f"- {translate_driver(lang, item)}")
        st.markdown("</div>", unsafe_allow_html=True)

    missing_features = prediction.get("missing_features") or []
    extra_features = prediction.get("extra_features") or []
    if missing_features or extra_features:
        with st.expander(text(lang, "result.coverage"), expanded=False):
            if missing_features:
                st.caption(text(lang, "result.missing_features").format(count=len(missing_features)))
            if extra_features:
                st.caption(text(lang, "result.extra_features").format(features=", ".join(extra_features)))


def render_single_client(api_base_url: str, lang: str) -> None:
    """Single-client scoring page."""

    render_header(text(lang, "page.single.title"), text(lang, "page.single.subtitle"))
    submitted, values = collect_single_client_values(lang)

    if submitted:
        payload = {"features": build_feature_payload(values)}
        prediction, error = post_json(f"{api_base_url}/predict", payload)
        st.session_state["single_prediction"] = prediction
        st.session_state["single_error"] = error
        st.session_state["single_values"] = values

    render_prediction_result(
        st.session_state.get("single_prediction"),
        st.session_state.get("single_error"),
        st.session_state.get("single_values", values),
        lang,
    )


def risk_level_from_probability(probability: float, lang: str = "en") -> str:
    """Map probability to display risk level."""

    if probability < 0.2:
        return text(lang, "batch.low")
    if probability < 0.5:
        return text(lang, "batch.medium")
    return text(lang, "batch.high")


def render_batch_scoring(api_base_url: str, lang: str) -> None:
    """CSV batch scoring page."""

    render_header(text(lang, "page.batch.title"), text(lang, "page.batch.subtitle"))
    uploaded_file = st.file_uploader(text(lang, "batch.upload"), type=["csv"])
    if uploaded_file is None:
        st.markdown(f'<div class="callout">{text(lang, "batch.callout")}</div>', unsafe_allow_html=True)
        return

    frame = pd.read_csv(uploaded_file)
    st.subheader(text(lang, "batch.preview"))
    st.dataframe(frame.head(10), width="stretch")

    if st.button(text(lang, "batch.run"), width="stretch"):
        items = [{"features": build_feature_payload(row.dropna().to_dict())} for _, row in frame.iterrows()]
        response, error = post_json(f"{api_base_url}/predict-batch", {"items": items}, timeout=30.0)
        if error:
            st.error(error)
        else:
            predictions = response.get("predictions", []) if response else []
            result = frame.copy()
            result["default_probability"] = [item["default_probability"] for item in predictions]
            result["risk_level"] = [
                risk_level_from_probability(float(item["default_probability"]), lang) for item in predictions
            ]
            st.session_state["batch_result"] = result

    result = st.session_state.get("batch_result")
    if isinstance(result, pd.DataFrame):
        st.subheader(text(lang, "batch.results"))
        result = result.copy()
        result["risk_level"] = [
            risk_level_from_probability(float(probability), lang) for probability in result["default_probability"]
        ]
        result_columns = [column for column in ["SK_ID_CURR", "default_probability", "risk_level"] if column in result.columns]
        st.dataframe(result[result_columns].head(30), width="stretch")
        st.download_button(
            text(lang, "batch.download"),
            data=result.to_csv(index=False).encode("utf-8"),
            file_name="home_credit_predictions.csv",
            mime="text/csv",
            width="stretch",
        )


def render_data_explorer(lang: str) -> None:
    """Data explorer page based on committed report artifacts."""

    render_header(text(lang, "page.data.title"), text(lang, "page.data.subtitle"))
    images = [
        (text(lang, "data.target_distribution"), "target_distribution.png"),
        (text(lang, "data.top_missing_values"), "missingness_top20.png"),
        (text(lang, "data.credit_risk_correlations"), "target_correlations_top20.png"),
        (text(lang, "data.pca_projection"), "pca_projection.png"),
    ]
    first, second = st.columns(2)
    for index, (caption, filename) in enumerate(images):
        path = IMAGE_DIR / filename
        with first if index % 2 == 0 else second:
            if path.exists():
                st.image(str(path), caption=caption, width="stretch")
            else:
                st.warning(text(lang, "data.missing_artifact").format(path=path))

    missing = load_missing_values_preview()
    if not missing.empty:
        st.subheader(text(lang, "data.missing_values_table"))
        st.dataframe(missing, width="stretch")


def render_model_explainability(lang: str) -> None:
    """Model explainability page."""

    render_header(text(lang, "page.explain.title"), text(lang, "page.explain.subtitle"))
    features = load_selected_features()
    if features.empty:
        st.warning(text(lang, "explain.feature_unavailable"))
    else:
        top = features.head(20)
        st.subheader(text(lang, "explain.global_importance"))
        st.bar_chart(top.set_index("feature")["importance"], horizontal=True)
        st.dataframe(top[["rank", "feature", "importance"]], width="stretch")

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.markdown(f"### {text(lang, 'result.increasing')}")
            st.write(f"+ {translate_driver(lang, 'Low EXT_SOURCE_2 increases risk')}")
            st.write(f"+ {translate_driver(lang, 'High annuity / income ratio increases risk')}")
            st.write(f"+ {translate_driver(lang, 'High credit / income ratio increases risk')}")
    with right:
        with st.container(border=True):
            st.markdown(f"### {text(lang, 'result.reducing')}")
            st.write(f"- {translate_driver(lang, 'Strong external score decreases risk')}")
            st.write(f"- {translate_driver(lang, 'Stable employment decreases risk')}")
            st.write(f"- {translate_driver(lang, 'Moderate annuity burden decreases risk')}")


def render_model_performance(lang: str) -> None:
    """Model performance page."""

    render_header(text(lang, "page.performance.title"), text(lang, "page.performance.subtitle"))
    results = load_experiment_results()
    if results.empty:
        st.warning(text(lang, "performance.results_unavailable"))
        return

    best = results.sort_values("test_roc_auc", ascending=False).iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ROC-AUC", f"{best['test_roc_auc']:.3f}")
    col2.metric("Precision", f"{best['test_precision']:.3f}")
    col3.metric("Recall", f"{best['test_recall']:.3f}")
    col4.metric("F1-score", f"{best['test_f1']:.3f}")

    st.subheader(text(lang, "performance.experiment_table"))
    display_columns = [
        "model",
        "test_roc_auc",
        "test_average_precision",
        "test_f1",
        "test_precision",
        "test_recall",
        "test_accuracy",
    ]
    st.dataframe(results[display_columns], width="stretch")

    st.subheader(text(lang, "performance.threshold_analysis"))
    threshold = st.slider(text(lang, "performance.decision_threshold"), 0.05, 0.80, 0.35, 0.01)
    st.markdown(
        f"""
        <div class="callout">
            <strong>{text(lang, "performance.current_threshold")}: {threshold:.2f}</strong><br>
            {text(lang, "performance.threshold_text")}
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Run the Streamlit app."""

    st.set_page_config(
        page_title="Home Credit Risk Scoring",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    page, api_base_url, _, lang = render_sidebar(os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL))

    if page == "overview":
        render_overview(lang)
    elif page == "single_client":
        render_single_client(api_base_url, lang)
    elif page == "batch_scoring":
        render_batch_scoring(api_base_url, lang)
    elif page == "data_explorer":
        render_data_explorer(lang)
    elif page == "model_explainability":
        render_model_explainability(lang)
    elif page == "model_performance":
        render_model_performance(lang)


if __name__ == "__main__":
    main()
