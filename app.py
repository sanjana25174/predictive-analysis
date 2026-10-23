"""
app.py - Smart Manufacturing Dashboard (Streamlit user interface).

All machine-learning and data logic lives in core.py. This file only draws pages.
Run with:   streamlit run app.py
"""

from __future__ import annotations

import html
import platform
from datetime import datetime

import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import sklearn
import streamlit as st
from plotly.subplots import make_subplots

import core

# set_page_config must be the first Streamlit command in the script.
st.set_page_config(
    page_title="Smart Manufacturing Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Design tokens
# --------------------------------------------------------------------------
NAVY = "#0F1F38"
INK = "#1B2A3D"
MUTED = "#5B6B80"
GRID = "#E3E8EF"
TEAL = "#0E9F8E"
BLUE = "#2563EB"
AMBER = "#D98A00"
RED = "#C8372D"
FONT = '"Segoe UI", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif'

OUTCOME_ORDER = ["No failure", "Failure"]
OUTCOME_COLORS = {"No failure": TEAL, "Failure": RED}
# risk level -> (text colour, tint background, accent colour, symbol). Text is dark for readability.
RISK_STYLE = {
    "Lower": ("#0B6E63", "#D8F1EE", TEAL, "✔"),
    "Medium": ("#7A4E00", "#FBEBC8", AMBER, "▲"),
    "Higher": ("#9B2118", "#F6D5D2", RED, "⚠"),
    "Not available": (MUTED, "#E8ECF2", MUTED, "•"),
}
RISK_COLORS = {level: style[2] for level, style in RISK_STYLE.items()}

PAGES = [
    ("Overview Dashboard", "📊"),
    ("Predict Machine Failure", "🔍"),
    ("Machine Health", "🩺"),
    ("Model Performance", "🎯"),
    ("Sensor Analytics", "📈"),
    ("Prediction History", "🕘"),
    ("About Project", "ℹ️"),
]

INPUT_KEYS = {
    "Air temperature [K]": "in_air",
    "Process temperature [K]": "in_process",
    "Rotational speed [rpm]": "in_rpm",
    "Torque [Nm]": "in_torque",
    "Tool wear [min]": "in_wear",
}
INT_INPUTS = {"Rotational speed [rpm]", "Tool wear [min]"}
PERSISTED_KEYS = list(INPUT_KEYS.values()) + ["in_type", "thr_med", "thr_high", "in_product_id"]

# Sample scenarios for demonstration. Each one is sent through the real model;
# the dashboard never assumes or displays a predetermined result for them.
SAMPLES = {
    "Typical readings": {
        "help": "Readings close to the dataset medians.",
        "values": {"Air temperature [K]": 300.1, "Process temperature [K]": 310.1,
                   "Rotational speed [rpm]": 1503, "Torque [Nm]": 40.1, "Tool wear [min]": 108},
        "type": "L",
    },
    "High tool wear": {
        "help": "Tool wear near the top of the dataset range; other readings typical.",
        "values": {"Air temperature [K]": 300.0, "Process temperature [K]": 310.0,
                   "Rotational speed [rpm]": 1500, "Torque [Nm]": 45.0, "Tool wear [min]": 235},
        "type": "L",
    },
    "High torque and wear": {
        "help": "Higher torque combined with high tool wear and slightly lower speed.",
        "values": {"Air temperature [K]": 300.0, "Process temperature [K]": 310.0,
                   "Rotational speed [rpm]": 1400, "Torque [Nm]": 62.0, "Tool wear [min]": 215},
        "type": "L",
    },
    "Narrow temperature gap": {
        "help": "Smaller gap between air and process temperature, with lower speed.",
        "values": {"Air temperature [K]": 302.5, "Process temperature [K]": 310.8,
                   "Rotational speed [rpm]": 1330, "Torque [Nm]": 45.0, "Tool wear [min]": 100},
        "type": "L",
    },
}

CSS = """
<style>
.stApp { background: #F3F5F8; font-family: "Segoe UI", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1320px; }
header[data-testid="stHeader"] { background: transparent; }

/* Sidebar */
section[data-testid="stSidebar"] { background: #0F1F38; }
section[data-testid="stSidebar"] * { color: #D5DEEB; }
section[data-testid="stSidebar"] .stButton > button {
  width: 100%; justify-content: flex-start; text-align: left; background: transparent;
  border: 0; border-radius: 10px; padding: 0.55rem 0.8rem; font-weight: 500; color: #C9D6E8;
}
section[data-testid="stSidebar"] .stButton > button > div { width: 100%; justify-content: flex-start; }
section[data-testid="stSidebar"] .stButton > button p { text-align: left; color: inherit; }
section[data-testid="stSidebar"] .stButton > button:hover { background: #1B3254; color: #FFFFFF; }
section[data-testid="stSidebar"] .stButton > button[kind="primary"],
section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] {
  background: #1B3254; color: #FFFFFF; box-shadow: inset 3px 0 0 #19C3AE;
}
section[data-testid="stSidebar"] .brand-title { color: #FFFFFF; font-size: 1.15rem; font-weight: 650; line-height: 1.25; }
section[data-testid="stSidebar"] .brand-sub { color: #8FA3C0; font-size: 0.82rem; margin-top: 2px; }
section[data-testid="stSidebar"] .side-note { color: #8FA3C0; font-size: 0.78rem; line-height: 1.45; }
section[data-testid="stSidebar"] .status-chip {
  display: inline-block; background: #12304F; border: 1px solid #1F4A73; border-radius: 999px;
  padding: 2px 10px; font-size: 0.78rem; color: #A8E6DD;
}

/* Page header */
.page-header { display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; margin-bottom: 1rem; }
.ph-title { font-size: 1.85rem; font-weight: 650; color: #1B2A3D; line-height: 1.2; }
.ph-sub { color: #5B6B80; margin-top: 0.25rem; font-size: 1rem; }
.date-chip { background: #FFFFFF; border: 1px solid #E3E8EF; border-radius: 999px; padding: 0.35rem 0.9rem;
  color: #5B6B80; font-size: 0.88rem; white-space: nowrap; }

/* Cards */
div[class*="st-key-card_"] {
  background: #FFFFFF; border: 1px solid #E3E8EF !important; border-radius: 14px;
  box-shadow: 0 1px 2px rgba(15,31,56,0.06), 0 6px 18px rgba(15,31,56,0.05); padding: 0.35rem 0.5rem;
}
.section-title { font-size: 1.08rem; font-weight: 650; color: #1B2A3D; margin: 0.2rem 0 0.1rem 0; }

/* KPI cards */
.kpi { background: #FFFFFF; border-radius: 14px; padding: 0.95rem 1.1rem; border: 1px solid #E3E8EF;
  border-top: 4px solid var(--accent, #0E9F8E); box-shadow: 0 1px 2px rgba(15,31,56,0.06), 0 6px 18px rgba(15,31,56,0.05);
  height: 100%; }
.kpi-label { color: #5B6B80; font-size: 0.88rem; }
.kpi-value { color: #1B2A3D; font-size: 1.95rem; font-weight: 650; line-height: 1.25; }
.kpi-note { color: #5B6B80; font-size: 0.8rem; }
.kpi-small .kpi-value { font-size: 1.5rem; }

/* Banners */
.banner { border-radius: 10px; padding: 0.7rem 1rem; margin: 0.2rem 0 1rem 0; font-size: 0.93rem; line-height: 1.5; }
.banner-info { background: #E8F0FE; border-left: 4px solid #2563EB; color: #1B2A3D; }
.banner-warn { background: #FFF4DB; border-left: 4px solid #D98A00; color: #1B2A3D; }

/* Prediction result */
.result-badge { display: inline-block; border-radius: 999px; padding: 0.3rem 0.95rem; font-weight: 650; font-size: 1rem; }
.result-row { display: flex; justify-content: space-between; padding: 0.45rem 0; border-bottom: 1px solid #E3E8EF; font-size: 0.95rem; }
.result-row span:first-child { color: #5B6B80; }
.result-row span:last-child { color: #1B2A3D; font-weight: 600; text-align: right; }
.reco { border-radius: 10px; padding: 0.8rem 1rem; margin-top: 0.8rem; border-left: 5px solid var(--accent, #0E9F8E);
  background: var(--tint, #D8F1EE); color: #1B2A3D; }
.reco b { display: block; margin-bottom: 2px; }
.placeholder { text-align: center; color: #5B6B80; padding: 3.2rem 1rem; }
.placeholder .big { font-size: 2.2rem; }

/* Architecture flow */
.flow { display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem; margin: 0.4rem 0 1rem 0; }
.flow-step { border-radius: 10px; padding: 0.55rem 0.85rem; font-size: 0.9rem; font-weight: 600; border: 1px solid; }
.flow-train { background: #E8F0FE; border-color: #B7CDF8; color: #17347A; }
.flow-predict { background: #D8F1EE; border-color: #9BD8D0; color: #0B5A51; }
.flow-arrow { color: #5B6B80; font-weight: 700; }
</style>
"""

# --------------------------------------------------------------------------
# Small UI helpers
# --------------------------------------------------------------------------
esc = html.escape


def pct(value: float, digits: int = 2) -> str:
    return f"{100 * value:.{digits}f}%"


def kpi_card(label: str, value: str, note: str = "", accent: str = TEAL, small: bool = False) -> str:
    size_class = " kpi-small" if small else ""
    return (
        f'<div class="kpi{size_class}" style="--accent:{accent}">'
        f'<div class="kpi-label">{esc(label)}</div>'
        f'<div class="kpi-value">{esc(value)}</div>'
        f'<div class="kpi-note">{esc(note)}</div></div>'
    )


def banner(text_html: str, kind: str = "info") -> None:
    """text_html is trusted text written in this file (may contain <b>)."""
    st.markdown(f'<div class="banner banner-{kind}">{text_html}</div>', unsafe_allow_html=True)


def card(name: str):
    """A white rounded container. 'key' lets the CSS style it; older Streamlit versions ignore it."""
    try:
        return st.container(border=True, key=f"card_{name}")
    except TypeError:
        return st.container(border=True)


def section(title: str, caption: str | None = None) -> None:
    st.markdown(f'<div class="section-title">{esc(title)}</div>', unsafe_allow_html=True)
    if caption:
        st.caption(caption)


def page_header(title: str, subtitle: str) -> None:
    today = datetime.now().strftime("%A, %d %B %Y")
    st.markdown(
        f'<div class="page-header"><div><div class="ph-title">{esc(title)}</div>'
        f'<div class="ph-sub">{esc(subtitle)}</div></div>'
        f'<div class="date-chip">{esc(today)}</div></div>',
        unsafe_allow_html=True,
    )


def show_chart(fig, key: str) -> None:
    """Draw a Plotly chart full width (works on both newer and older Streamlit versions)."""
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except Exception:  # noqa: BLE001 - older Streamlit uses use_container_width
        st.plotly_chart(fig, use_container_width=True, key=key)


def show_table(data: pd.DataFrame) -> None:
    try:
        st.dataframe(data, width="stretch", hide_index=True)
    except Exception:  # noqa: BLE001
        st.dataframe(data, use_container_width=True, hide_index=True)


def style_fig(fig, height: int = 330):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=8, r=8, t=36, b=8),
        font=dict(family=FONT, size=13, color=INK),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, title_text=""),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


# --------------------------------------------------------------------------
# Cached loading (the model is loaded once, never retrained)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the saved Random Forest model...")
def get_assets():
    return core.load_model_assets()


@st.cache_data(show_spinner="Loading the dataset...")
def get_dataset() -> pd.DataFrame:
    return core.load_dataset()


@st.cache_data(show_spinner="Loading the original dataset with Product IDs...")
def get_original_dataset() -> pd.DataFrame:
    """Load raw AI4I rows so a Product ID can retrieve its recorded sensor values."""
    path = core.BASE_DIR / "ai4i2020.csv"
    if not path.exists():
        raise core.AssetError(
            "The original dataset file 'ai4i2020.csv' was not found in the project folder. "
            "Keep it beside app.py to select Product IDs."
        )
    raw = pd.read_csv(path)
    required = ["Product ID", "Type", *core.SENSOR_COLUMNS]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise core.AssetError(
            "The original ai4i2020.csv is missing required columns: " + ", ".join(missing)
        )
    return raw


def apply_product_record() -> None:
    """Copy the selected original dataset record's sensors into the prediction form."""
    product_id = st.session_state.get("selected_product_id")
    if not product_id:
        st.session_state["in_product_id"] = ""
        return

    raw = get_original_dataset()
    matches = raw[raw["Product ID"].astype(str) == str(product_id)]
    if matches.empty:
        st.session_state["in_product_id"] = ""
        return

    row = matches.iloc[0]
    for column, key in INPUT_KEYS.items():
        value = row[column]
        st.session_state[key] = int(value) if column in INT_INPUTS else float(value)
    st.session_state["in_type"] = str(row["Type"])
    st.session_state["in_product_id"] = str(row["Product ID"])


@st.cache_data(show_spinner="Evaluating the saved model on the held-out test set...")
def get_evaluation() -> dict:
    model, features = get_assets()
    return core.evaluate_saved_model(model, features, get_dataset())


# --------------------------------------------------------------------------
# Session state and callbacks
# --------------------------------------------------------------------------
def init_state(df: pd.DataFrame) -> None:
    ss = st.session_state
    defaults = core.default_inputs(df)
    ss.setdefault("page", PAGES[0][0])
    ss.setdefault("history", [])
    ss.setdefault("last_result", None)
    ss.setdefault("last_product_id", "")
    ss.setdefault("in_product_id", "")
    ss.setdefault("selected_product_id", None)
    ss.setdefault("confirm_clear", False)
    ss.setdefault("thr_med", core.DEFAULT_RISK_THRESHOLDS[0])
    ss.setdefault("thr_high", core.DEFAULT_RISK_THRESHOLDS[1])
    for column, key in INPUT_KEYS.items():
        ss.setdefault(key, defaults[column])
    ss.setdefault("in_type", defaults["Machine type"])
    # Streamlit forgets a widget's value when its page is not drawn; re-saving keeps the form filled.
    for key in PERSISTED_KEYS:
        ss[key] = ss[key]


def go_to(page: str) -> None:
    st.session_state["page"] = page


def apply_sample(name: str) -> None:
    sample = SAMPLES[name]
    for column, key in INPUT_KEYS.items():
        value = sample["values"][column]
        st.session_state[key] = int(value) if column in INT_INPUTS else float(value)
    st.session_state["in_type"] = sample["type"]
    # Sample scenarios are manual examples, not a selected Product ID record.
    st.session_state["in_product_id"] = ""
    st.session_state["selected_product_id"] = None


def reset_thresholds() -> None:
    st.session_state["thr_med"], st.session_state["thr_high"] = core.DEFAULT_RISK_THRESHOLDS


def ask_clear_history() -> None:
    st.session_state["confirm_clear"] = True


def cancel_clear_history() -> None:
    st.session_state["confirm_clear"] = False


def confirm_clear_history() -> None:
    st.session_state["history"] = []
    st.session_state["confirm_clear"] = False


def current_thresholds() -> tuple[float, float]:
    return float(st.session_state["thr_med"]), float(st.session_state["thr_high"])


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            '<div class="brand-title">⚙️ Predictive Maintenance</div>'
            '<div class="brand-sub">AI-based smart manufacturing</div>',
            unsafe_allow_html=True,
        )
        st.write("")
        for index, (name, icon) in enumerate(PAGES):
            active = st.session_state["page"] == name
            st.button(
                f"{icon}  {name}",
                key=f"nav_{index}",
                on_click=go_to,
                args=(name,),
                type="primary" if active else "secondary",
            )
        st.write("")
        st.markdown('<span class="status-chip">✔ Model and dataset loaded</span>', unsafe_allow_html=True)
        st.write("")
        st.markdown(
            '<div class="side-note">Prototype for demonstration. Uses the public AI4I 2020 dataset and '
            "manual sensor input. Not validated on any factory machine and not a safety system.</div>",
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------
# Page 1 - Overview
# --------------------------------------------------------------------------
def page_overview(ctx: dict) -> None:
    df = ctx["df"]
    evaluation = get_evaluation()
    overview = core.dataset_overview(df)

    page_header(
        "Smart Manufacturing Dashboard",
        "AI-powered machine failure prediction and maintenance decision support.",
    )
    banner(
        "<b>Historical dataset information.</b> The record counts and sensor statistics on this page "
        "describe the public AI4I 2020 dataset used to train the model. They are not live factory "
        "measurements, and the records are not live machines."
    )

    kpis = st.columns(4)
    kpis[0].markdown(
        kpi_card("Total dataset records", f"{overview['total']:,}", "Historical AI4I 2020 records", BLUE),
        unsafe_allow_html=True,
    )
    kpis[1].markdown(
        kpi_card("Failure cases in dataset", f"{overview['failures']:,}",
                 f"{overview['failure_pct']:.2f}% of historical records", RED),
        unsafe_allow_html=True,
    )
    kpis[2].markdown(
        kpi_card("Non-failure cases in dataset", f"{overview['non_failures']:,}",
                 f"{overview['non_failure_pct']:.2f}% of historical records", TEAL),
        unsafe_allow_html=True,
    )
    kpis[3].markdown(
        kpi_card("Model accuracy", pct(evaluation["accuracy"]),
                 f"Random Forest, held-out test set ({evaluation['n_test']:,} records)", NAVY),
        unsafe_allow_html=True,
    )
    st.write("")

    left, right = st.columns([3, 2], gap="large")
    with left:
        with card("overview_distribution"):
            section("Failure distribution", "Historical records by outcome (AI4I 2020 dataset)")
            data = pd.DataFrame(
                {
                    "Outcome": OUTCOME_ORDER,
                    "Records": [overview["non_failures"], overview["failures"]],
                    "Share": [overview["non_failure_pct"], overview["failure_pct"]],
                }
            )
            data["Label"] = [f"{r:,} ({s:.2f}%)" for r, s in zip(data["Records"], data["Share"])]
            fig = px.bar(data, x="Outcome", y="Records", color="Outcome", text="Label",
                         color_discrete_map=OUTCOME_COLORS, category_orders={"Outcome": OUTCOME_ORDER})
            fig.update_traces(textposition="outside", cliponaxis=False,
                              hovertemplate="%{x}<br>%{y:,} records<extra></extra>")
            fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Records",
                              yaxis_range=[0, overview["non_failures"] * 1.15])
            show_chart(style_fig(fig, 330), "ov_failure_distribution")

    with right:
        with card("overview_model_summary"):
            section("Model summary", "Random Forest on the held-out test set")
            row1 = st.columns(2)
            row1[0].markdown(kpi_card("Accuracy", pct(evaluation["accuracy"]), "", TEAL, small=True), unsafe_allow_html=True)
            row1[1].markdown(kpi_card("Precision", pct(evaluation["precision"]), "", BLUE, small=True), unsafe_allow_html=True)
            st.write("")
            row2 = st.columns(2)
            row2[0].markdown(kpi_card("Recall", pct(evaluation["recall"]), "", AMBER, small=True), unsafe_allow_html=True)
            row2[1].markdown(kpi_card("F1-score", pct(evaluation["f1"]), "", NAVY, small=True), unsafe_allow_html=True)
            st.markdown(
                f"**Why accuracy alone is not enough.** Only about {overview['failure_pct']:.2f}% of records are "
                f"failures. On this test set, a model that always answered \"no failure\" would already be "
                f"{pct(evaluation['baseline_accuracy'])} accurate while catching none of the "
                f"{evaluation['n_test_failures']} real failures. Recall shows how many real failures are caught "
                f"({pct(evaluation['recall'], 1)}); precision shows how many alarms are genuine "
                f"({pct(evaluation['precision'], 1)}); F1-score balances the two."
            )

    with card("overview_sensors"):
        section("Sensor overview", "Summary statistics and distributions of the historical dataset (all 10,000 records)")
        show_table(core.sensor_summary(df).round(2))
        for start in (0, 3):
            cols = st.columns(3)
            for slot, column in zip(cols, core.SENSOR_COLUMNS[start:start + 3]):
                with slot:
                    st.markdown(f"**{core.sensor_name(column)}**")
                    fig = px.histogram(df, x=column, nbins=30, color_discrete_sequence=[BLUE])
                    fig.update_layout(bargap=0.05, showlegend=False, xaxis_title=column, yaxis_title="Records")
                    show_chart(style_fig(fig, 240), f"ov_hist_{column}")
            if start == 3:
                with cols[2]:
                    st.markdown("**Machine type mix**")
                    mix = (df[core.TYPE_LABEL_COLUMN].value_counts().reindex(core.MACHINE_TYPES)
                           .rename_axis("Machine type").reset_index(name="Records"))
                    fig = px.bar(mix, x="Machine type", y="Records", color_discrete_sequence=[NAVY], text="Records")
                    fig.update_traces(textposition="outside", cliponaxis=False)
                    fig.update_layout(showlegend=False, xaxis_title="Machine type (L / M / H)", yaxis_title="Records")
                    show_chart(style_fig(fig, 240), "ov_type_mix")

    with card("overview_quick_action"):
        section("Quick action", "Estimate failure risk for a set of sensor readings you enter.")
        st.button("Open Predict Machine Failure", type="primary", key="qa_predict",
                  on_click=go_to, args=("Predict Machine Failure",))


# --------------------------------------------------------------------------
# Page 2 - Predict Machine Failure
# --------------------------------------------------------------------------
def gauge_figure(probability: float, thresholds: tuple[float, float], accent: str):
    low, high = thresholds
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=100 * probability,
            number={"suffix": "%", "valueformat": ".1f", "font": {"size": 42, "color": INK}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": accent, "thickness": 0.28},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 100 * low], "color": "#D8F1EE"},
                    {"range": [100 * low, 100 * high], "color": "#FBEBC8"},
                    {"range": [100 * high, 100], "color": "#F6D5D2"},
                ],
            },
        )
    )
    fig.update_layout(height=220, margin=dict(l=24, r=24, t=10, b=0),
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(family=FONT, color=INK))
    return fig


def render_prediction_result(
    result: core.PredictionResult,
    current_inputs: dict,
    current_type: str,
    prediction_product_id: str,
    current_product_id: str,
) -> None:
    text_color, tint, accent, symbol = RISK_STYLE.get(result.risk_level, RISK_STYLE["Not available"])
    label = core.RISK_INFO[result.risk_level]["label"]
    low, high = result.thresholds

    section("Prediction result", f"Run at {result.timestamp} using the saved Random Forest model")
    if result.probability is not None:
        show_chart(gauge_figure(result.probability, result.thresholds, accent), "pred_gauge")
        probability_text = f"{100 * result.probability:.1f}%"
    else:
        probability_text = "Not available"

    st.markdown(
        f'<span class="result-badge" style="background:{tint};color:{text_color};border:1px solid {accent}">'
        f"{symbol} {esc(label)}</span>",
        unsafe_allow_html=True,
    )
    rows = [
        ("Product ID (dataset record)", esc(prediction_product_id or "Manual sensor input")),
        ("Estimated failure probability", probability_text),
        ("Predicted class", f"{esc(result.class_label)} ({result.predicted_class})"),
        ("Risk level", esc(label)),
        ("Machine type", esc(result.machine_type)),
    ]
    st.markdown(
        "".join(f'<div class="result-row"><span>{name}</span><span>{value}</span></div>' for name, value in rows),
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="reco" style="--accent:{accent};--tint:{tint}"><b>Maintenance recommendation</b>'
        f"{esc(result.recommendation)}</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Predicted class uses the model's own {core.CLASS_DECISION_THRESHOLD:.0%} decision rule. The risk level uses "
        f"prototype bands: below {low:.0%} lower, {low:.0%} to below {high:.0%} medium, {high:.0%} and above higher. "
        "These bands are prototype settings, not industrially validated safety limits."
    )

    entered = pd.DataFrame(
        {
            "Sensor": [core.sensor_name(c) for c in core.SENSOR_COLUMNS],
            "Entered value": [f"{result.inputs[c]:g}" for c in core.SENSOR_COLUMNS],
            "Unit": [core.sensor_unit(c) for c in core.SENSOR_COLUMNS],
        }
    )
    st.markdown("**Entered sensor values**")
    show_table(entered)

    st.info(
        "This result is a statistical estimate from a model trained on a public dataset. It is not a "
        "guarantee that a machine will or will not fail, and it does not replace inspection by qualified staff."
    )
    with st.expander("Show the encoded row sent to the model"):
        st.caption("Columns are in exactly the order the model was trained with (from feature_names.pkl).")
        st.dataframe(result.model_input.T.rename(columns={0: "Value"}))

    if (
        current_inputs != result.inputs
        or current_type != result.machine_type
        or current_product_id != prediction_product_id
    ):
        st.warning("The selected Product ID or sensor inputs have changed since this prediction. Click Predict to update the result.")


def page_predict(ctx: dict) -> None:
    df, model, features = ctx["df"], ctx["model"], ctx["features"]
    ranges = core.observed_ranges(df)

    page_header(
        "Predict Machine Failure",
        "Enter sensor readings to estimate failure probability, risk level and a maintenance recommendation.",
    )

    left, right = st.columns([5, 6], gap="large")

    with left:
        with card("predict_inputs"):
            section("Sensor input panel")
            st.caption(
                "Choose a Product ID from the original AI4I 2020 dataset to automatically fill its recorded "
                "sensor readings and machine type. You can still edit the values before predicting."
            )
            try:
                original_df = get_original_dataset()
                product_ids = original_df["Product ID"].astype(str).drop_duplicates().tolist()
                st.selectbox(
                    "Select Product ID from the dataset",
                    options=product_ids,
                    index=None,
                    placeholder="Search or choose a Product ID",
                    key="selected_product_id",
                    on_change=apply_product_record,
                    help="Selecting a Product ID loads its historical sensor readings. Product ID itself is not sent to the model.",
                )
                if st.session_state.get("in_product_id"):
                    st.success(
                        f"Loaded historical sensor readings for Product ID {st.session_state['in_product_id']}. "
                        "The model predicts from the sensor values, not from the Product ID."
                    )
            except (core.AssetError, OSError, pd.errors.ParserError) as exc:
                st.warning(f"Product ID selection is unavailable: {exc}. You can still enter sensor readings manually.")

            st.caption("Sample scenarios for demonstration only. Each one is run through the real model; "
                       "no result is predetermined.")
            names = list(SAMPLES)
            for pair in (names[:2], names[2:]):
                cols = st.columns(2)
                for slot, name in zip(cols, pair):
                    with slot:
                        st.button(name, key=f"sample_{name}", on_click=apply_sample, args=(name,),
                                  help=SAMPLES[name]["help"])
            st.divider()

            def range_help(column: str) -> str:
                low, high = ranges[column]
                return (f"Observed dataset range: {low:g} to {high:g} {core.sensor_unit(column)}. "
                        "Values outside this range are allowed but may be outside the model's training distribution.")

            c1, c2 = st.columns(2)
            with c1:
                st.number_input("Air temperature [K]", min_value=0.0, step=0.1, format="%.1f",
                                key="in_air", help=range_help("Air temperature [K]"))
                st.number_input("Rotational speed [rpm]", min_value=1, step=10,
                                key="in_rpm", help=range_help("Rotational speed [rpm]"))
                st.number_input("Tool wear [min]", min_value=0, step=1,
                                key="in_wear", help=range_help("Tool wear [min]"))
            with c2:
                st.number_input("Process temperature [K]", min_value=0.0, step=0.1, format="%.1f",
                                key="in_process", help=range_help("Process temperature [K]"))
                st.number_input("Torque [Nm]", min_value=0.0, step=0.1, format="%.1f",
                                key="in_torque", help=range_help("Torque [Nm]"))
                st.selectbox("Machine type", core.MACHINE_TYPES, key="in_type",
                             help="Product quality variant used in the dataset: L (low), M (medium) or H (high).")

            values = {column: st.session_state[key] for column, key in INPUT_KEYS.items()}
            machine_type = st.session_state["in_type"]
            errors, warnings_ = core.validate_inputs(values, machine_type, ranges)
            for message in errors:
                st.error(message)
            for message in warnings_:
                st.warning(message)

            with st.expander("Risk thresholds (prototype settings)"):
                st.caption("These bands come from this project's original script. They are prototype settings, "
                           "not industrially validated safety limits. Changing them changes only the risk label, "
                           "not the model's probability.")
                t1, t2 = st.columns(2)
                t1.slider("Medium risk starts at", 0.05, 0.95, step=0.05, format="%.2f", key="thr_med")
                t2.slider("Higher risk starts at", 0.05, 0.95, step=0.05, format="%.2f", key="thr_high")
                st.button("Reset to project defaults", key="reset_thr", on_click=reset_thresholds)
            thresholds = current_thresholds()
            thresholds_ok = thresholds[0] < thresholds[1]
            if not thresholds_ok:
                st.error("'Medium risk starts at' must be lower than 'Higher risk starts at'.")

            clicked = st.button("Predict machine failure", type="primary", key="predict_btn",
                                disabled=bool(errors) or not thresholds_ok)
            if clicked:
                try:
                    result = core.predict_failure(model, features, values, machine_type, thresholds)
                except (ValueError, core.AssetError) as exc:
                    st.error(f"Prediction could not be completed: {exc}")
                else:
                    st.session_state["last_result"] = result
                    st.session_state["last_product_id"] = st.session_state.get("in_product_id", "")
                    history_row = result.to_history_row()
                    history_row["Product ID"] = st.session_state["last_product_id"] or "Manual sensor input"
                    st.session_state["history"].append(history_row)

    with right:
        with card("predict_result"):
            result = st.session_state["last_result"]
            if result is None:
                st.markdown(
                    '<div class="placeholder"><div class="big">🔍</div>'
                    "<b>No prediction yet</b><br>Enter sensor readings (or load a sample scenario) and click "
                    "<b>Predict machine failure</b>. The estimate comes from your saved Random Forest model."
                    "</div>",
                    unsafe_allow_html=True,
                )
            else:
                render_prediction_result(
                    result,
                    {c: values[c] for c in core.SENSOR_COLUMNS},
                    machine_type,
                    st.session_state.get("last_product_id", ""),
                    st.session_state.get("in_product_id", ""),
                )


# --------------------------------------------------------------------------
# Shared chart builders (Machine Health and Sensor Analytics)
# --------------------------------------------------------------------------
def outcome_histogram(data: pd.DataFrame, column: str, bins: int, height: int = 300):
    fig = px.histogram(data, x=column, color=core.OUTCOME_COLUMN, nbins=bins, barmode="overlay",
                       opacity=0.65, histnorm="percent", color_discrete_map=OUTCOME_COLORS,
                       category_orders={core.OUTCOME_COLUMN: OUTCOME_ORDER})
    fig.update_layout(bargap=0.05, xaxis_title=column, yaxis_title="Share of each group (%)")
    return style_fig(fig, height)


def outcome_box(data: pd.DataFrame, column: str, height: int = 300):
    fig = px.box(data, x=core.OUTCOME_COLUMN, y=column, color=core.OUTCOME_COLUMN, points="outliers",
                 color_discrete_map=OUTCOME_COLORS, category_orders={core.OUTCOME_COLUMN: OUTCOME_ORDER})
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title=column)
    return style_fig(fig, height)


def failure_rate_figure(table: pd.DataFrame, column: str, height: int = 300):
    fig = px.bar(table, x="Range", y="Failure rate (%)", hover_data={"Records": True, "Failures": True},
                 category_orders={"Range": list(table["Range"])}, color_discrete_sequence=[BLUE])
    fig.update_layout(xaxis_title=f"{column} (equal-sized bins)", yaxis_title="Observed failure rate (%)")
    fig.update_xaxes(tickangle=-35)
    return style_fig(fig, height)


# --------------------------------------------------------------------------
# Page 3 - Machine Health (historical data)
# --------------------------------------------------------------------------
def page_machine_health(ctx: dict) -> None:
    df = ctx["df"]
    page_header("Machine Health",
                "How sensor readings relate to machine failure in the historical training dataset.")
    banner(
        "<b>Historical analysis only.</b> This page studies recorded values from the public AI4I 2020 dataset. "
        "The dashboard is not connected to real machines or real-time sensors, so nothing here shows the "
        "current condition of any actual machine."
    )

    with card("health_summary"):
        section("Sensor summary statistics", "All 10,000 historical records")
        show_table(core.sensor_summary(df).round(2))

    left, right = st.columns(2, gap="large")
    differences = core.standardized_differences(df)
    with left:
        with card("health_means"):
            section("Failure vs non-failure records", "Average sensor value in each group")
            table = differences[["Sensor", "Unit", "Mean (no failure)", "Mean (failure)", "Difference"]].round(2)
            show_table(table)
    with right:
        with card("health_std_diff"):
            section("Size of the difference",
                    "Difference in group means divided by the overall standard deviation. Descriptive only.")
            fig = px.bar(differences, x="Standardized difference", y="Sensor", orientation="h",
                         color_discrete_sequence=[BLUE],
                         hover_data={"Mean (no failure)": ":.2f", "Mean (failure)": ":.2f", "Unit": True})
            fig.add_vline(x=0, line_color=MUTED)
            fig.update_layout(xaxis_title="Standardized difference (failure minus no failure)", yaxis_title="")
            show_chart(style_fig(fig, 280), "health_std_diff_chart")

    with card("health_explorer"):
        section("Explore one sensor", "Pick a sensor to see how its distribution and failure rate compare.")
        column = st.selectbox("Sensor", core.SENSOR_COLUMNS, key="mh_sensor")
        a, b = st.columns(2, gap="large")
        with a:
            st.markdown("**Distribution by outcome**")
            st.caption("Each group is scaled to 100% so the shapes can be compared despite the imbalance.")
            show_chart(outcome_histogram(df, column, 30), "mh_hist")
        with b:
            st.markdown("**Spread by outcome**")
            st.caption("Box plots show the median, quartiles and outliers.")
            show_chart(outcome_box(df, column), "mh_box")
        c, d = st.columns(2, gap="large")
        with c:
            st.markdown("**Failure rate across the sensor's range**")
            st.caption("Records are split into equal-sized bins; each bar is the share of failures in that bin.")
            show_chart(failure_rate_figure(core.failure_rate_by_bin(df, column), column), "mh_rate")
        with d:
            st.markdown("**Failure rate by machine type**")
            st.caption("Share of failure records within each product quality variant.")
            by_type = (df.groupby(core.TYPE_LABEL_COLUMN)[core.TARGET_COLUMN].agg(["size", "sum"])
                       .reindex(core.MACHINE_TYPES))
            type_table = pd.DataFrame(
                {"Machine type": by_type.index, "Records": by_type["size"].to_numpy(),
                 "Failures": by_type["sum"].to_numpy().astype(int)})
            type_table["Failure rate (%)"] = 100 * type_table["Failures"] / type_table["Records"]
            fig = px.bar(type_table, x="Machine type", y="Failure rate (%)", color_discrete_sequence=[NAVY],
                         hover_data={"Records": True, "Failures": True},
                         category_orders={"Machine type": core.MACHINE_TYPES})
            fig.update_layout(xaxis_title="Machine type (L / M / H)", yaxis_title="Observed failure rate (%)")
            show_chart(style_fig(fig, 300), "mh_type_rate")

    with card("health_limits"):
        section("How to read this page")
        st.markdown(
            "- These are recorded values from a public dataset, **not live readings** from any machine.\n"
            "- Differences between groups and bar heights describe the data; they **do not prove what causes failures**.\n"
            "- Failures are rare (about 3.4%), so bins with few records give noisy failure rates.\n"
            "- A real machine can differ from this dataset, so these patterns may not carry over to a particular factory."
        )


# --------------------------------------------------------------------------
# Page 4 - Model Performance
# --------------------------------------------------------------------------
def confusion_figure(ev: dict):
    counts = np.array([[ev["tn"], ev["fp"]], [ev["fn"], ev["tp"]]])
    share = counts / counts.sum(axis=1, keepdims=True)
    labels = [[f"{counts[i][j]:,}<br>{share[i][j]:.1%} of actual" for j in range(2)] for i in range(2)]
    fig = go.Figure(
        go.Heatmap(z=share, x=["Predicted: No failure", "Predicted: Failure"],
                   y=["Actual: No failure", "Actual: Failure"], text=labels, texttemplate="%{text}",
                   colorscale=[[0, "#EEF2F7"], [1, BLUE]], zmin=0, zmax=1, showscale=False, hoverinfo="skip")
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(template="plotly_white", height=330, margin=dict(l=8, r=8, t=16, b=8),
                      font=dict(family=FONT, size=14, color=INK),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def page_model_performance(ctx: dict) -> None:
    df, model, features = ctx["df"], ctx["model"], ctx["features"]
    ev = get_evaluation()
    saved = core.load_saved_metrics()
    overview = core.dataset_overview(df)

    page_header("Model Performance", "Evaluation of the three compared models on a held-out test set.")
    banner(
        f"<b>Held-out test set.</b> Results come from {ev['n_test']:,} records ({ev['n_test_failures']} failures) "
        "that were not used to train the models (20% stratified split, random_state 42). They describe "
        "performance on this public dataset and are not a guarantee of future industrial performance."
    )

    kpis = st.columns(4)
    for slot, (label, key, accent) in zip(kpis, [("Accuracy", "accuracy", TEAL), ("Precision", "precision", BLUE),
                                                 ("Recall", "recall", AMBER), ("F1-score", "f1", NAVY)]):
        slot.markdown(kpi_card(f"Random Forest {label.lower()}", pct(ev[key]), "Recomputed from the saved model", accent),
                      unsafe_allow_html=True)
    st.write("")

    # ---- Verification -------------------------------------------------------------------
    with card("perf_verification"):
        section("Verification of the reported results",
                "The saved random_forest_model.pkl was evaluated again on the reproduced test split.")
        check = core.compare_with_reported(ev)
        shown = check.copy()
        shown["Matches"] = np.where(shown["Matches"], "✔ Yes", "✘ No")
        show_table(shown)
        if check["Matches"].all():
            st.success("Verified: the saved model reproduces every previously reported figure.")
        else:
            st.warning("Some previously reported figures do not match the saved model. "
                       "Use the recomputed values and update your report.")

    # ---- Model comparison ---------------------------------------------------------------
    with card("perf_comparison"):
        section("Model comparison", "Test-set accuracy, precision, recall and F1-score")
        if saved is None:
            st.warning("model_metrics.json was not found, so only the Random Forest is verified (above). "
                       "Logistic Regression and Decision Tree values need to be recalculated. "
                       "Run `python evaluate_models.py` in the project folder, then reload this page.")
            comparison = {"Random Forest": {k: ev[k] for k in ("accuracy", "precision", "recall", "f1")}}
        else:
            comparison = {name: res["test"] for name, res in saved["models"].items()}
            rf_saved = saved["models"]["Random Forest"]["test"]
            if any(abs(rf_saved[k] - ev[k]) > 1e-9 for k in ("accuracy", "precision", "recall", "f1")):
                st.warning("model_metrics.json no longer matches the saved model. "
                           "Run `python evaluate_models.py` to refresh it.")
        rows = []
        for model_name in ["Logistic Regression", "Decision Tree", "Random Forest"]:
            if model_name in comparison:
                for metric, key in [("Accuracy", "accuracy"), ("Precision", "precision"),
                                    ("Recall", "recall"), ("F1-score", "f1")]:
                    rows.append({"Model": model_name, "Metric": metric, "Score (%)": 100 * comparison[model_name][key]})
        long = pd.DataFrame(rows)
        fig = px.bar(long, x="Metric", y="Score (%)", color="Model", barmode="group", text_auto=".1f",
                     color_discrete_map={"Logistic Regression": "#9AA9BD", "Decision Tree": BLUE, "Random Forest": TEAL},
                     category_orders={"Metric": ["Accuracy", "Precision", "Recall", "F1-score"]})
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(yaxis_range=[0, 112], xaxis_title="", yaxis_title="Score (%)")
        show_chart(style_fig(fig, 380), "perf_comparison_chart")

        wide = long.pivot(index="Model", columns="Metric", values="Score (%)")
        wide = wide.reindex(columns=["Accuracy", "Precision", "Recall", "F1-score"]).round(2).reset_index()
        show_table(wide)

        status = [{"Model": "Random Forest", "Source of test metrics": "Saved .pkl evaluated live in this app",
                   "Status": "Verified against reported figures" if check["Matches"].all() else "Differs from reported figures"}]
        for name in ("Logistic Regression", "Decision Tree"):
            if saved is None:
                status.append({"Model": name, "Source of test metrics": "Not available", "Status": "Needs recalculation"})
            else:
                status.append({"Model": name, "Source of test metrics": saved["models"][name]["source"],
                               "Status": f"Recomputed by evaluate_models.py ({saved['generated_at']}, scikit-learn {saved['sklearn_version']})"})
        show_table(pd.DataFrame(status))

        if saved is not None:
            best = {m: max(comparison, key=lambda name: comparison[name][k])
                    for m, k in [("accuracy", "accuracy"), ("precision", "precision"), ("recall", "recall"), ("F1-score", "f1")]}
            st.markdown(
                f"**Reading the comparison.** Highest accuracy: {best['accuracy']}. Highest precision: {best['precision']}. "
                f"Highest recall: {best['recall']}. Highest F1-score: {best['F1-score']}. A model with higher recall "
                "catches more real failures but usually raises more false alarms, so the best choice depends on how "
                "costly a missed failure is compared with an unnecessary inspection."
            )

    # ---- Confusion matrix -----------------------------------------------------------------
    with card("perf_confusion"):
        section("Random Forest confusion matrix", "Each cell shows the count and its share of the actual class")
        a, b = st.columns([3, 2], gap="large")
        with a:
            show_chart(confusion_figure(ev), "perf_confusion_chart")
        with b:
            actual_failures = ev["tp"] + ev["fn"]
            alarms = ev["tp"] + ev["fp"]
            st.markdown(
                f"- **True negatives ({ev['tn']:,}):** healthy records correctly left alone.\n"
                f"- **False positives ({ev['fp']:,}):** false alarms. The model flagged a failure that did not happen, "
                "which could mean an unnecessary inspection.\n"
                f"- **False negatives ({ev['fn']:,}):** missed failures. A real failure the model did not flag, "
                "which is the costliest error.\n"
                f"- **True positives ({ev['tp']:,}):** real failures correctly flagged.\n\n"
                f"The model caught {ev['tp']} of {actual_failures} real failures (recall {pct(ev['recall'], 1)}). "
                f"Of {alarms} alarms, {ev['tp']} were genuine (precision {pct(ev['precision'], 1)})."
            )

    # ---- Imbalance ---------------------------------------------------------------------------
    with card("perf_imbalance"):
        section("Why dataset imbalance matters")
        st.markdown(
            f"The dataset is about **{overview['non_failure_pct']:.2f}% non-failure** and "
            f"**{overview['failure_pct']:.2f}% failure** ({overview['non_failures']:,} vs {overview['failures']:,} records). "
            f"A model that always predicted \"no failure\" would score {pct(ev['baseline_accuracy'])} accuracy on this test set "
            "while finding no failures at all. That is why precision, recall and F1-score are reported next to accuracy, "
            "and why the training script uses balanced class weights."
        )

    # ---- Threshold explorer ------------------------------------------------------------------
    with card("perf_threshold"):
        section("Decision threshold explorer",
                "How precision and recall trade off on the test set as the failure-probability cut-off moves")
        threshold = st.slider("Flag a record as failure when estimated probability is at least", 0.05, 0.95, 0.50,
                              step=0.05, format="%.2f", key="mp_threshold")
        tm = core.threshold_metrics(ev["y_true"], ev["proba"], threshold)
        cols = st.columns(4)
        cols[0].markdown(kpi_card("Precision", pct(tm["precision"], 1), "Share of alarms that were real", BLUE, small=True), unsafe_allow_html=True)
        cols[1].markdown(kpi_card("Recall", pct(tm["recall"], 1), "Share of real failures caught", AMBER, small=True), unsafe_allow_html=True)
        cols[2].markdown(kpi_card("False alarms", f"{tm['fp']:,}", "False positives", RED, small=True), unsafe_allow_html=True)
        cols[3].markdown(kpi_card("Missed failures", f"{tm['fn']:,}", "False negatives", NAVY, small=True), unsafe_allow_html=True)
        sweep = core.threshold_sweep(ev["y_true"], ev["proba"])
        fig = px.line(sweep, x="Threshold", y=["Precision", "Recall", "F1-score"], markers=True,
                      color_discrete_sequence=[BLUE, AMBER, TEAL])
        fig.add_vline(x=threshold, line_dash="dash", line_color=MUTED)
        fig.update_layout(xaxis_title="Decision threshold", yaxis_title="Score", yaxis_tickformat=".0%")
        show_chart(style_fig(fig, 320), "perf_threshold_chart")
        st.caption("The Predict page's own class output uses the 0.50 rule, which is the first row of numbers above "
                   "when the slider is at 0.50.")

    # ---- Risk bands on held-out data ---------------------------------------------------------
    with card("perf_bands"):
        section("How the prototype risk bands behave on test data",
                "Observed failure rate among held-out test records that fall in each band")
        thresholds = current_thresholds()
        if thresholds[0] >= thresholds[1]:
            thresholds = core.DEFAULT_RISK_THRESHOLDS
            st.caption("The current Predict page thresholds are invalid, so the project defaults are shown.")
        bands = core.risk_band_table(ev["y_true"], ev["proba"], thresholds).round(2)
        show_table(bands)
        st.caption(f"Bands in use: medium starts at {thresholds[0]:.0%}, higher starts at {thresholds[1]:.0%} "
                   "(change them on the Predict page). They are prototype settings. This table shows how they behave "
                   "on this dataset only; it does not validate them for any real machine.")

    # ---- Feature importance ----------------------------------------------------------------
    with card("perf_importance"):
        section("Which inputs the Random Forest relies on",
                "Impurity-based feature importance from the saved model")
        importance = core.feature_importance_table(model, features)
        if importance.empty:
            st.caption("This model does not provide feature importances.")
        else:
            fig = px.bar(importance.iloc[::-1], x="Importance", y="Feature", orientation="h",
                         color_discrete_sequence=[TEAL])
            fig.update_layout(xaxis_title="Importance (sums to 1)", yaxis_title="")
            show_chart(style_fig(fig, 330), "perf_importance_chart")
            st.caption("Importance describes what the model used to separate the training records. "
                       "It does not show that an input causes failures.")

    # ---- Train vs test -------------------------------------------------------------------------
    if saved is not None:
        with st.expander("Training vs test performance (overfitting check)"):
            rows = []
            for name, res in saved["models"].items():
                rows.append({"Model": name,
                             "Train accuracy (%)": round(100 * res["train"]["accuracy"], 2),
                             "Test accuracy (%)": round(100 * res["test"]["accuracy"], 2),
                             "Train recall (%)": round(100 * res["train"]["recall"], 2),
                             "Test recall (%)": round(100 * res["test"]["recall"], 2),
                             "Train precision (%)": round(100 * res["train"]["precision"], 2),
                             "Test precision (%)": round(100 * res["test"]["precision"], 2)})
            show_table(pd.DataFrame(rows))
            st.caption("A large gap between training and test scores suggests the model fits the training records more "
                       "closely than new records. The test-set figures are the ones to quote.")


# --------------------------------------------------------------------------
# Page 5 - Sensor Analytics
# --------------------------------------------------------------------------
def page_sensor_analytics(ctx: dict) -> None:
    df = ctx["df"]
    page_header("Sensor Analytics", "Interactive charts built from the historical AI4I 2020 dataset.")

    wear_col = "Tool wear [min]"
    wear_min, wear_max = int(df[wear_col].min()), int(df[wear_col].max())
    with card("analytics_filters"):
        section("Filters", "Filters apply to every chart on this page.")
        f1, f2, f3, f4 = st.columns([2, 2, 3, 2])
        types = f1.multiselect("Machine type", core.MACHINE_TYPES, default=core.MACHINE_TYPES, key="sa_types")
        outcome_choice = f2.selectbox("Records to include", ["All records", "Failure only", "No failure only"],
                                      key="sa_outcome")
        wear_range = f3.slider("Tool wear range [min]", wear_min, wear_max, (wear_min, wear_max), key="sa_wear")
        bins = f4.slider("Histogram bins", 10, 80, 30, key="sa_bins")

    data = df[df[core.TYPE_LABEL_COLUMN].isin(types)]
    data = data[data[wear_col].between(wear_range[0], wear_range[1])]
    if outcome_choice == "Failure only":
        data = data[data[core.TARGET_COLUMN] == 1]
    elif outcome_choice == "No failure only":
        data = data[data[core.TARGET_COLUMN] == 0]

    if data.empty:
        st.warning("No records match the current filters. Widen the filters to see charts.")
        return
    both = data[core.TARGET_COLUMN].nunique() == 2
    st.caption(f"Showing {len(data):,} of {len(df):,} historical records "
               f"({int(data[core.TARGET_COLUMN].sum()):,} failures).")

    with card("analytics_distributions"):
        section("Sensor distributions", "Histogram and box plot for each sensor")
        tabs = st.tabs([core.sensor_name(c) for c in core.SENSOR_COLUMNS])
        for tab, column in zip(tabs, core.SENSOR_COLUMNS):
            with tab:
                a, b = st.columns(2, gap="large")
                with a:
                    if both:
                        st.caption("Each outcome group is scaled to 100% so shapes can be compared.")
                        fig = outcome_histogram(data, column, bins)
                    else:
                        st.caption("Count of records.")
                        fig = px.histogram(data, x=column, nbins=bins, color_discrete_sequence=[BLUE])
                        fig.update_layout(bargap=0.05, showlegend=False, xaxis_title=column, yaxis_title="Records")
                        style_fig(fig, 300)
                    show_chart(fig, f"sa_hist_{column}")
                with b:
                    st.caption("Median, quartiles and outliers.")
                    show_chart(outcome_box(data, column), f"sa_box_{column}")

    with card("analytics_relationships"):
        section("Sensor relationships", "Scatter plot of two sensors. Failure records are drawn on top.")
        s1, s2 = st.columns(2)
        x_col = s1.selectbox("X axis", core.SENSOR_COLUMNS, index=2, key="sa_x")
        y_col = s2.selectbox("Y axis", core.SENSOR_COLUMNS, index=3, key="sa_y")
        if x_col == y_col:
            st.info("Choose two different sensors to compare.")
        else:
            fig = px.scatter(data, x=x_col, y=y_col, color=core.OUTCOME_COLUMN, opacity=0.55,
                             color_discrete_map=OUTCOME_COLORS, category_orders={core.OUTCOME_COLUMN: OUTCOME_ORDER})
            fig.update_layout(xaxis_title=x_col, yaxis_title=y_col)
            show_chart(style_fig(fig, 420), "sa_scatter")

    with card("analytics_correlation"):
        section("Correlation heatmap",
                "Pearson correlation between sensors and the failure flag (1 = failure, 0 = no failure)")
        corr_columns = list(core.SENSOR_COLUMNS)
        if both:
            corr_columns.append(core.TARGET_COLUMN)
        else:
            st.caption("Only one outcome is present after filtering, so the failure flag is left out.")
        corr = data[corr_columns].corr()
        short = [core.sensor_name(c) if c != core.TARGET_COLUMN else "Machine failure (0/1)" for c in corr_columns]
        corr.index, corr.columns = short, short
        fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r", aspect="auto")
        fig.update_layout(coloraxis_colorbar=dict(title="r"))
        show_chart(style_fig(fig, 420), "sa_corr")
        st.caption("Correlation measures how two columns move together. It does not show that one causes the other, "
                   "and it only captures straight-line relationships.")

    with card("analytics_compare"):
        section("Failure vs non-failure comparison", "Box plots of each sensor, split by outcome")
        if not both:
            st.info("Both outcomes are needed for this comparison. Set 'Records to include' to All records.")
        else:
            fig = make_subplots(rows=1, cols=5,
                                subplot_titles=[f"{core.sensor_name(c)} [{core.sensor_unit(c)}]" for c in core.SENSOR_COLUMNS])
            for position, column in enumerate(core.SENSOR_COLUMNS, start=1):
                for outcome in OUTCOME_ORDER:
                    part = data[data[core.OUTCOME_COLUMN] == outcome]
                    if part.empty:
                        continue
                    fig.add_trace(go.Box(y=part[column], name=outcome, marker_color=OUTCOME_COLORS[outcome],
                                         legendgroup=outcome, showlegend=(position == 1), boxpoints=False),
                                  row=1, col=position)
            fig.update_annotations(font_size=11)
            style_fig(fig, 400)
            show_chart(fig, "sa_compare")

    st.caption("All charts describe historical records, not live machines. Patterns in this dataset may not apply to a particular factory.")


# --------------------------------------------------------------------------
# Page 6 - Prediction History
# --------------------------------------------------------------------------
def page_history(ctx: dict) -> None:
    page_header("Prediction History", "Predictions you triggered on the Predict Machine Failure page.")
    history = st.session_state["history"]
    banner(
        "<b>History is temporary.</b> Records are kept only in this browser session. They are cleared when you "
        "refresh the page, close the tab or restart the app, and nothing is saved to disk. "
        "Use the download button if you want to keep a copy. Rows here are predictions made by this app, "
        "never rows from the historical dataset.",
        "warn",
    )

    if not history:
        with card("history_empty"):
            st.markdown("**No predictions yet.** Run a prediction and it will appear here.")
            st.button("Go to Predict Machine Failure", type="primary", key="hist_go_predict",
                      on_click=go_to, args=("Predict Machine Failure",))
        return

    table = pd.DataFrame(history)
    probabilities = pd.to_numeric(table["Failure probability (%)"], errors="coerce")
    kpis = st.columns(3)
    kpis[0].markdown(kpi_card("Predictions this session", f"{len(table)}", "Triggered by you", BLUE), unsafe_allow_html=True)
    kpis[1].markdown(kpi_card("Average estimated probability", f"{probabilities.mean():.1f}%", "Across this session's predictions", TEAL), unsafe_allow_html=True)
    kpis[2].markdown(kpi_card("Medium or higher risk", f"{int(table['Risk level'].isin(['Medium', 'Higher']).sum())}",
                              "Using the risk bands in force at each prediction", AMBER), unsafe_allow_html=True)
    st.write("")

    if len(table) >= 2:
        with card("history_chart"):
            section("Estimated failure probability per prediction", "Oldest on the left, newest on the right")
            chart_data = table.copy()
            chart_data["Prediction #"] = range(1, len(chart_data) + 1)
            fig = px.bar(chart_data, x="Prediction #", y="Failure probability (%)", color="Risk level",
                         color_discrete_map=RISK_COLORS, hover_data=["Timestamp", "Machine type"])
            fig.update_layout(xaxis_title="Prediction number", yaxis_title="Estimated failure probability (%)",
                              yaxis_range=[0, 100], xaxis=dict(dtick=1))
            show_chart(style_fig(fig, 300), "history_chart_fig")

    with card("history_table"):
        section("Prediction records", "Newest first")
        show_table(table.iloc[::-1].reset_index(drop=True))
        st.download_button("Download history as CSV", data=table.to_csv(index=False).encode("utf-8"),
                           file_name="prediction_history.csv", mime="text/csv", key="history_download")

    with card("history_clear"):
        section("Clear history")
        if st.session_state["confirm_clear"]:
            st.warning(f"This will permanently delete all {len(table)} prediction records from this session. "
                       "This cannot be undone.")
            c1, c2 = st.columns(2)
            c1.button("Yes, delete the history", type="primary", key="clear_yes", on_click=confirm_clear_history)
            c2.button("Cancel", key="clear_no", on_click=cancel_clear_history)
        else:
            st.button("Clear history", key="clear_ask", on_click=ask_clear_history)


# --------------------------------------------------------------------------
# Page 7 - About Project
# --------------------------------------------------------------------------
def flow_html(steps: list[str], kind: str) -> str:
    arrow = '<span class="flow-arrow">→</span>'
    return '<div class="flow">' + arrow.join(f'<div class="flow-step flow-{kind}">{esc(s)}</div>' for s in steps) + "</div>"


def page_about(ctx: dict) -> None:
    df, model, features = ctx["df"], ctx["model"], ctx["features"]
    overview = core.dataset_overview(df)
    page_header("About Project", "AI-Based Smart Manufacturing & Predictive Maintenance System")

    with card("about_problem"):
        section("Problem statement")
        st.markdown(
            "Unplanned machine failures cause downtime, repair costs and safety risk, while fixed-interval "
            "maintenance can waste effort on machines that are healthy. Maintenance teams need an early, "
            "data-driven indication of which operating conditions look risky."
        )
        section("Proposed solution")
        st.markdown(
            "Train classification models on historical sensor data, keep the best one (Random Forest), and expose it "
            "through a dashboard. The user enters sensor readings and receives an estimated failure probability, a "
            "risk level and a suggested maintenance action. The output supports human decision-making; it does not "
            "control any machine."
        )

    with card("about_dataset"):
        section("Dataset description")
        st.markdown(
            f"**AI4I 2020 Predictive Maintenance Dataset**, a public dataset that simulates industrial machine data. "
            f"It contains {overview['total']:,} records ({overview['non_failure_pct']:.2f}% non-failure, "
            f"{overview['failure_pct']:.2f}% failure).\n\n"
            "- **Model inputs:** air temperature [K], process temperature [K], rotational speed [rpm], torque [Nm], "
            "tool wear [min], and machine type (L, M or H, one-hot encoded into Type_L, Type_M, Type_H).\n"
            "- **Target:** machine failure (0 = no failure, 1 = failure).\n"
            "- **Product ID:** used only to select a historical row and load its recorded sensor values in the Predict page. "
            "It is not a model input and does not directly determine the prediction.\n"
            "- **Left out of the model:** record identifiers and the individual failure-mode flags, because the flags "
            "directly define the target and would leak the answer."
        )

    a, b = st.columns(2, gap="large")
    with a:
        with card("about_stack"):
            section("Technology stack")
            st.markdown(
                "- Python 3.12\n- Pandas and NumPy for data handling\n- Scikit-learn for the models and metrics\n"
                "- Joblib to save and load the trained model\n- Streamlit for the dashboard\n- Plotly for interactive charts"
            )
    with b:
        with card("about_models"):
            section("ML models compared")
            st.markdown(
                "All three used an 80/20 stratified split with random_state 42.\n"
                "- **Logistic Regression:** scaled inputs, balanced class weights\n"
                "- **Decision Tree:** max depth 5, balanced class weights\n"
                "- **Random Forest (final model):** 200 trees, max depth 10, balanced class weights"
            )

    with card("about_architecture"):
        section("System architecture")
        st.markdown("**Training time** (run offline with data_cleaning.py and model_training.py, not when you use the dashboard)")
        st.markdown(flow_html(["Historical Sensor Dataset", "Data Cleaning", "Feature Encoding",
                               "Model Training and Evaluation", "Saved Random Forest Model"], "train"),
                    unsafe_allow_html=True)
        st.markdown("**Prediction time** (runs in app.py and core.py every time you click Predict; the model is only loaded, never retrained)")
        st.markdown(flow_html(["Saved Random Forest Model", "Streamlit Dashboard", "User Sensor Inputs",
                               "Failure Probability", "Risk Level", "Maintenance Recommendation"], "predict"),
                    unsafe_allow_html=True)

    c, d = st.columns(2, gap="large")
    with c:
        with card("about_limitations"):
            section("Current limitations")
            st.markdown(
                "- Uses a public dataset.\n- Requires manual sensor input.\n"
                "- Has not been validated on a particular factory's machines.\n"
                "- May produce false alarms or miss actual failures.\n"
                "- Risk thresholds are prototype settings unless validated separately."
            )
    with d:
        with card("about_future"):
            section("Future scope")
            st.markdown(
                "- Real-time IoT or PLC integration\n- Machine-specific model training\n"
                "- Better false-alarm management\n- Maintenance history\n- Secure database integration\n"
                "- Production deployment after appropriate validation"
            )

    with card("about_status"):
        section("System status", "Useful when something does not look right")
        trained_with = core.model_sklearn_version(model)
        runtime = sklearn.__version__
        status = [
            ("Model file", f"{core.MODEL_PATH.name} ({type(model).__name__}, {getattr(model, 'n_estimators', '?')} trees)"),
            ("Model features (in order)", ", ".join(features)),
            ("Dataset file", f"{core.DATA_PATH.name} ({overview['total']:,} records)"),
            ("scikit-learn: model saved with", trained_with or "unknown"),
            ("scikit-learn: running now", runtime),
            ("Python / Streamlit / Plotly", f"{platform.python_version()} / {st.__version__} / {plotly.__version__}"),
        ]
        show_table(pd.DataFrame(status, columns=["Item", "Value"]))
        if trained_with and trained_with != runtime:
            st.warning(f"The model was saved with scikit-learn {trained_with} but this app runs {runtime}. "
                       "It usually still works, but the safest fix is to install the same version "
                       f"(pip install scikit-learn=={trained_with}) or re-run model_training.py.")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
PAGE_FUNCTIONS = {
    "Overview Dashboard": page_overview,
    "Predict Machine Failure": page_predict,
    "Machine Health": page_machine_health,
    "Model Performance": page_model_performance,
    "Sensor Analytics": page_sensor_analytics,
    "Prediction History": page_history,
    "About Project": page_about,
}


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    try:
        model, features = get_assets()
        df = get_dataset()
    except core.AssetError as exc:
        st.error(str(exc))
        st.info("Put app.py, core.py, ai4i2020_cleaned.csv, random_forest_model.pkl and feature_names.pkl in the "
                "same folder, then run `streamlit run app.py` from that folder.")
        st.stop()

    init_state(df)
    render_sidebar()
    context = {"df": df, "model": model, "features": features}
    try:
        PAGE_FUNCTIONS[st.session_state["page"]](context)
    except Exception as exc:  # noqa: BLE001 - show a readable message instead of a blank page
        st.error("Something went wrong while drawing this page. Details are shown below.")
        st.exception(exc)


main()
