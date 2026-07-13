"""
Streamlit interface for the Decolonial Appropriate Technology (DAT) framework.

Run with:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running `streamlit run app/streamlit_app.py` straight from a repo clone
# without requiring `pip install -e .` first.
_APP_DIR = Path(__file__).resolve().parent
SRC_DIR = _APP_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dat_framework.config import (
    DOMAIN_WEIGHT_PRESETS,
    FairnessWeights,
    PROTECTED_ATTRIBUTES,
    SUB_SAHARAN_COUNTRIES,
    UTILITY_RETENTION_FLOOR,
    WORLD_BANK_DEFAULT_INDICATORS,
    WORLD_BANK_INDICATOR_CATEGORIES,
    WORLD_BANK_INDICATORS,
)
from dat_framework.data.synthetic_dhs import SyntheticDataConfig, generate_dataset
from dat_framework.metrics.fairness_metrics import (
    intersectional_metrics,
    single_attribute_metrics,
)
from dat_framework.metrics.stats_tests import factorial_anova, non_additivity_check, paired_ttests
from dat_framework.optimization.moo_engine import run_w0_sweep
from dat_framework.optimization.pareto import (
    extract_pareto_front,
    recommend_solution,
    solutions_to_dataframe,
)
from dat_framework.optimization.preprocessing import assign_intersectional_subgroups

from ui_icons import icon_html

_FAVICON = _APP_DIR / "assets" / "favicon.svg"

st.set_page_config(
    page_title="DAT Framework — Ubuntu Fairness",
    layout="wide",
    page_icon=str(_FAVICON),
)

# ---------------------------------------------------------------------------
# Theme — earthy Ubuntu-inspired palette (deep teal + gold), consistent
# typography and spacing across every tab.
# ---------------------------------------------------------------------------
PALETTE = {
    "primary": "#0F6E63",      # deep teal
    "primary_dark": "#0A4F47",
    "accent": "#D98E2E",       # warm gold/ochre
    "danger": "#C0392B",
    "success": "#1E8449",
    "bg": "#F7F5F0",
    "card": "#FFFFFF",
    "text": "#20302B",
    "muted": "#4A5854",       # WCAG-friendly on #F7F5F0 (was too faint at #6B7A75)
    "colorway": ["#0F6E63", "#D98E2E", "#1E8449", "#C0392B", "#5B84B1", "#8E5572"],
}

NAV_SECTIONS = (
    {
        "id": "data",
        "label": "Data & context",
        "icon": ":material/database:",
        "description": "Synthetic cohort preview, attribute prevalence, and live World Bank indicators.",
    },
    {
        "id": "diagnostics",
        "label": "Bias diagnostics",
        "icon": ":material/search:",
        "description": "Single-attribute vs intersectional metrics — the mirage of neutrality.",
    },
    {
        "id": "statistics",
        "label": "Statistics",
        "icon": ":material/bar_chart:",
        "description": "t-tests, factorial ANOVA, and non-additivity spotlight.",
    },
    {
        "id": "moo",
        "label": "MOO framework",
        "icon": ":material/tune:",
        "description": "Tier 1–3 DAT workflow: mapping, optimization sweep, and Pareto governance.",
    },
)
_NAV_BY_ID = {section["id"]: section for section in NAV_SECTIONS}

st.markdown(
    f"""
    <style>
    :root {{
        --dat-text: {PALETTE['text']};
        --dat-text-muted: {PALETTE['muted']};
        --dat-bg: {PALETTE['bg']};
        --dat-card: {PALETTE['card']};
        --dat-primary: {PALETTE['primary']};
        --dat-primary-dark: {PALETTE['primary_dark']};
        --dat-accent: {PALETTE['accent']};
        --dat-border: #E3E0D8;
    }}

    /* Base — force light color scheme so Streamlit widgets stay readable */
    html, body {{
        color-scheme: light;
        font-family: "Source Sans Pro", "Segoe UI", sans-serif;
    }}
    .stApp {{
        background-color: var(--dat-bg);
        color: var(--dat-text);
    }}

    /* Main + sidebar: every standard text node Streamlit renders */
    [data-testid="stAppViewContainer"],
    [data-testid="stSidebarContent"],
    [data-testid="stMainBlockContainer"] {{
        color: var(--dat-text);
    }}
    [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] li,
    [data-testid="stAppViewContainer"] label,
    [data-testid="stAppViewContainer"] h1,
    [data-testid="stAppViewContainer"] h2,
    [data-testid="stAppViewContainer"] h3,
    [data-testid="stAppViewContainer"] h4,
    [data-testid="stAppViewContainer"] h5,
    [data-testid="stAppViewContainer"] h6,
    [data-testid="stAppViewContainer"] strong,
    [data-testid="stSidebarContent"] p,
    [data-testid="stSidebarContent"] label,
    [data-testid="stSidebarContent"] span,
    [data-testid="stWidgetLabel"] p,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] strong {{
        color: var(--dat-text) !important;
        -webkit-text-fill-color: var(--dat-text) !important;
        opacity: 1 !important;
    }}
    [data-testid="stCaptionContainer"] p,
    [data-testid="stCaptionContainer"] small {{
        color: var(--dat-text-muted) !important;
        -webkit-text-fill-color: var(--dat-text-muted) !important;
        opacity: 1 !important;
    }}

    /* Widget inputs — selected values, sliders, selects */
    [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stMultiSelect"] [data-baseweb="select"] > div,
    [data-baseweb="input"] input,
    [data-baseweb="textarea"] textarea,
    [data-baseweb="tag"] {{
        color: var(--dat-text) !important;
        -webkit-text-fill-color: var(--dat-text) !important;
    }}
    div[data-testid="stExpander"] summary,
    div[data-testid="stExpander"] summary span,
    div[data-testid="stExpander"] summary p {{
        color: var(--dat-text) !important;
        -webkit-text-fill-color: var(--dat-text) !important;
    }}

    .dat-hero {{
        background: linear-gradient(120deg, {PALETTE['primary_dark']} 0%, {PALETTE['primary']} 100%);
        border-radius: 14px;
        padding: 1.6rem 2rem;
        margin-bottom: 1.4rem;
        color: #FFFFFF;
        box-shadow: 0 4px 18px rgba(15, 110, 99, 0.25);
    }}
    .dat-hero h1 {{
        margin: 0 0 0.3rem 0;
        font-size: 1.9rem;
        font-weight: 700;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }}
    .dat-hero p {{
        margin: 0;
        font-size: 0.95rem;
        color: #E4F1EE !important;
        -webkit-text-fill-color: #E4F1EE !important;
        opacity: 0.95;
    }}
    .dat-section-title {{
        font-size: 1.05rem;
        font-weight: 700;
        color: {PALETTE['primary_dark']};
        border-left: 4px solid {PALETTE['accent']};
        padding-left: 0.6rem;
        margin: 0.4rem 0 0.8rem 0;
    }}
    .dat-card {{
        background-color: {PALETTE['card']};
        border: 1px solid var(--dat-border);
        border-radius: 10px;
        padding: 0.85rem 1rem 0.6rem 1rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        height: 100%;
    }}
    .dat-card-icon {{ margin-bottom: 4px; }}
    .dat-icon svg {{ display: block; }}
    .dat-inline-icon-row {{
        display: flex; align-items: flex-start; gap: 8px;
    }}
    .dat-inline-icon-row p {{
        margin: 0;
        color: var(--dat-text-muted) !important;
        -webkit-text-fill-color: var(--dat-text-muted) !important;
    }}
    .dat-card-label {{
        color: {PALETTE['muted']};
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }}
    .dat-card-value {{
        color: {PALETTE['text']};
        font-size: 1.6rem;
        font-weight: 800;
        margin: 2px 0;
        line-height: 1.1;
    }}
    .dat-card-sub {{ color: {PALETTE['muted']}; font-size: 0.75rem; }}

    /* Section nav — full-width segmented control with breathing room */
    div[data-testid="stSegmentedControl"] {{
        margin: 0.25rem 0 0.35rem 0;
    }}
    div[data-testid="stSegmentedControl"] [role="radiogroup"] {{
        flex-wrap: wrap;
        gap: 0.35rem;
    }}

    div[data-testid="stMetric"] {{
        background-color: {PALETTE['card']};
        border: 1px solid var(--dat-border);
        border-radius: 10px;
        padding: 0.85rem 1rem 0.6rem 1rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }}
    div[data-testid="stMetricLabel"] {{
        color: {PALETTE['muted']} !important;
    }}
    div[data-testid="stMetricValue"] {{
        color: {PALETTE['text']} !important;
    }}

    div[data-testid="stDataFrame"] {{
        border: 1px solid var(--dat-border);
        border-radius: 8px;
        overflow: hidden;
    }}

    /* Buttons — primary vs secondary (Reset, Pull live data, etc.) */
    .stButton > button[kind="primary"],
    .stButton > button[kind="primaryFormSubmit"] {{
        background-color: var(--dat-primary) !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        border-radius: 8px;
        border: none !important;
        font-weight: 600;
    }}
    .stButton > button[kind="primary"]:hover,
    .stButton > button[kind="primaryFormSubmit"]:hover {{
        background-color: var(--dat-primary-dark) !important;
        color: #FFFFFF !important;
    }}
    .stButton > button[kind="secondary"],
    .stButton > button:not([kind="primary"]):not([kind="primaryFormSubmit"]) {{
        background-color: #FFFFFF !important;
        color: var(--dat-primary-dark) !important;
        -webkit-text-fill-color: var(--dat-primary-dark) !important;
        border: 1px solid var(--dat-border) !important;
        border-radius: 8px;
        font-weight: 600;
    }}

    section[data-testid="stSidebar"] {{
        background-color: #EFEBE0;
    }}
    div[data-testid="stExpander"] {{
        background-color: {PALETTE['card']};
        border: 1px solid var(--dat-border) !important;
        border-radius: 10px;
        margin-bottom: 10px;
    }}
    div[data-testid="stAlert"] {{
        border-radius: 10px;
    }}
    div[data-testid="stAlert"] p {{
        color: var(--dat-text) !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def themed(fig: go.Figure, height: int | None = None) -> go.Figure:
    """Apply the app's consistent visual theme to a Plotly figure."""
    fig.update_layout(
        colorway=PALETTE["colorway"],
        font=dict(family="Source Sans Pro, Segoe UI, sans-serif", size=13, color=PALETTE["text"]),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=36, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    if height:
        fig.update_layout(height=height)
    tick_font = dict(color=PALETTE["text"], size=12)
    fig.update_xaxes(gridcolor="#EAE7DE", zerolinecolor="#EAE7DE", tickfont=tick_font)
    fig.update_yaxes(gridcolor="#EAE7DE", zerolinecolor="#EAE7DE", tickfont=tick_font)
    return fig


def metric_card_row(cards):
    cols = st.columns(len(cards))
    for col, (icon_name, label, value, sub) in zip(cols, cards):
        icon_markup = icon_html(icon_name, size=22, color=PALETTE["primary"]) if icon_name else ""
        with col:
            st.markdown(
                f"""
                <div class="dat-card">
                    <div class="dat-card-icon">{icon_markup}</div>
                    <div class="dat-card-label">{label}</div>
                    <div class="dat-card-value">{value}</div>
                    <div class="dat-card-sub">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Sidebar — data + weight controls
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    f"""
    <div style="display:flex; align-items:center; gap:8px; margin-bottom:2px;">
        {icon_html("globe", size=26, color="#0A4F47")}
        <span style="font-size:1.15rem; font-weight:800; color:#0A4F47;">DAT Ubuntu Fairness</span>
    </div>
    <div style="color:#4A5854; font-size:0.75rem; margin-bottom:14px;">v1.1.0</div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar.expander("1. Dataset Configuration", expanded=True):
    n_records = st.slider("Number of synthetic records", 500, 6000, 2000, step=250)
    seed = st.number_input("Random seed", value=42, step=1)
    min_group_size = st.slider("Minimum subgroup size (for metric stability)", 3, 30, 8)

with st.sidebar.expander("2. Fairness Weights (Tier 2)", expanded=True):
    domain = st.selectbox(
        "Domain preset", ["neutral", "health", "finance", "hiring", "custom"],
        help="Paper section 3.5: 'the weight of DIR may be 0.7 in a health dataset, "
             "and 0.3 in a finance dataset.' Hiring is weighted around DIR since the "
             "0.80 threshold this tool enforces is the legal four-fifths rule "
             "(EEOC Uniform Guidelines, 1978).",
    )
    if domain == "custom":
        w_dir = st.slider("w1 — DIR weight", 0.0, 1.0, 0.4)
        w_eod = st.slider("w2 — EOD weight", 0.0, 1.0 - w_dir, min(0.3, 1 - w_dir))
        w_dpd = round(1.0 - w_dir - w_eod, 4)
        st.caption(f"w3 — DPD weight (auto): {w_dpd}")
        weights = FairnessWeights(w_dir=w_dir, w_eod=w_eod, w_dpd=w_dpd)
    else:
        weights = DOMAIN_WEIGHT_PRESETS[domain]
        st.caption(f"w1(DIR)={weights.w_dir}, w2(EOD)={weights.w_eod}, w3(DPD)={weights.w_dpd}")

with st.sidebar.expander("3. Model & Evaluation", expanded=False):
    utility_floor_pct = st.slider(
        "Utility floor (% of baseline accuracy)", 80, 100, int(UTILITY_RETENTION_FLOOR * 100),
        help="Tier 2's gamma_utility constraint: the MOO sweep won't accept a solution "
             "that drops accuracy below this percentage of the baseline model's accuracy.",
    )
    slsqp_max_iter = st.slider(
        "SLSQP max iterations", 20, 150, 60, step=10,
        help="More iterations can find a better-optimized point per w0, at the cost of "
             "a slower sweep. 60 is a reasonable default for datasets under ~3,000 records.",
    )

st.sidebar.divider()
st.sidebar.caption(
    "Data note: DHS microdata is gated behind a registered-researcher account, so "
    "this app uses a structurally-equivalent **synthetic** stand-in calibrated to "
    "the paper's reported baseline bias magnitudes (tables 6.0/6.2). Swap in a real "
    "DHS extract by matching the schema in `data/synthetic_dhs.py`."
)


@st.cache_data(show_spinner=False)
def _get_data(n, seed_):
    df = generate_dataset(SyntheticDataConfig(n_records=n, seed=seed_))
    df = assign_intersectional_subgroups(df, PROTECTED_ATTRIBUTES)
    return df


df = _get_data(n_records, int(seed))

with st.sidebar.expander("4. Export & Reporting", expanded=False):
    st.download_button(
        "Export dataset (CSV)",
        data=df.drop(columns=["subgroup_tuple"]).to_csv(index=False).encode(),
        file_name="dat_synthetic_dataset.csv", mime="text/csv",
        use_container_width=True,
    )
    _report_single = single_attribute_metrics(df, y_pred_col="y_pred_baseline")
    _report_lines = [
        "DAT Ubuntu Fairness — Summary Report",
        f"Dataset: {len(df):,} synthetic records, seed={seed}, domain preset={domain}",
        "",
        "Single-attribute baseline DIR/DPD/EOD:",
        _report_single[["DIR", "DPD", "EOD"]].round(3).to_string(),
        "",
        f"Fairness weights: DIR={weights.w_dir}, EOD={weights.w_eod}, DPD={weights.w_dpd}",
        f"Minimum subgroup size: {min_group_size}",
    ]
    st.download_button(
        "Download report (TXT)",
        data="\n".join(_report_lines).encode(),
        file_name="dat_summary_report.txt", mime="text/plain",
        use_container_width=True,
    )

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
head_col, reset_col = st.columns([6, 1])
with head_col:
    st.markdown(
        f"""
        <div class="dat-hero">
            <h1 style="display:flex;align-items:center;gap:10px;margin:0 0 0.3rem 0;">
                {icon_html("globe", size=30, color="#FFFFFF")}
                From Universalism to Ubuntu
            </h1>
            <p>A working reproduction of the Decolonial Appropriate Technology (DAT) framework —
            intersectional fairness for AI/ML in Sub-Saharan Africa. Dube, Mguni &amp; Dube (2025), ICAT 2026.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with reset_col:
    st.write("")
    if st.button("Reset", use_container_width=True):
        st.cache_data.clear()
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

section = st.segmented_control(
    "Workflow section",
    options=[s["id"] for s in NAV_SECTIONS],
    format_func=lambda sid: f"{_NAV_BY_ID[sid]['icon']} {_NAV_BY_ID[sid]['label']}",
    default=NAV_SECTIONS[0]["id"],
    key="main_section",
    label_visibility="collapsed",
    width="stretch",
)
if section is None:
    section = NAV_SECTIONS[0]["id"]

st.caption(_NAV_BY_ID[section]["description"])
st.divider()


@st.cache_data(show_spinner=False)
def _get_worldbank_cached(countries: tuple, indicators: tuple):
    """Cached path used once we already know the pull succeeded."""
    from dat_framework.data.worldbank import fetch_all_indicators, latest_value_per_country
    countries_dict = {k: SUB_SAHARAN_COUNTRIES[k] for k in countries}
    indicators_dict = {k: WORLD_BANK_INDICATORS[k] for k in indicators}
    raw = fetch_all_indicators(countries_dict, indicators_dict)
    return latest_value_per_country(raw)


# ---------------------------------------------------------------------------
# TAB 1 — Data & Context
# ---------------------------------------------------------------------------
if section == "data":
    positive_rate = df["y_true"].mean()
    metric_card_row([
        ("database", "Dataset Size", f"{len(df):,}", "Synthetic Records"),
        ("users", "Protected Attributes", f"{len(PROTECTED_ATTRIBUTES)}", "Intersectional Groups"),
        ("chart", "Positive Rate (Overall)", f"{positive_rate:.0%}", "y_true = 1"),
        ("scale", "Fairness Domain", domain.capitalize(), "Domain Preset"),
        ("shield", "Minimum Group Size", f"{min_group_size}", "For Metric Stability"),
    ])
    st.write("")

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown('<div class="dat-section-title">Synthetic individual-level dataset</div>', unsafe_allow_html=True)
        st.dataframe(df.drop(columns=["subgroup_tuple"]).head(20), use_container_width=True)
        st.caption(f"{len(df):,} records · {len(PROTECTED_ATTRIBUTES)} protected attributes "
                   f"· {df['subgroup_id'].nunique()} intersectional subgroups populated "
                   f"(K = 2^{len(PROTECTED_ATTRIBUTES)} = {2**len(PROTECTED_ATTRIBUTES)} possible)")

    with col2:
        st.markdown('<div class="dat-section-title">Attribute prevalence</div>', unsafe_allow_html=True)
        prevalence = df[PROTECTED_ATTRIBUTES].mean().sort_values(ascending=True)
        fig = px.bar(
            prevalence, orientation="h",
            labels={"value": "Share of population marked disadvantaged", "index": ""},
            title=None,
            text=[f"{v:.0%}" for v in prevalence.values],
            color_discrete_sequence=[PALETTE["primary"]],
        )
        fig.update_traces(
            textposition="outside",
            textfont=dict(color=PALETTE["text"], size=12),
        )
        fig.update_layout(showlegend=False)
        themed(fig, height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown('<div class="dat-section-title">World Bank context (live pull)</div>', unsafe_allow_html=True)
    st.caption(
        "Section 3.2's country selection — one per Sub-Saharan sub-region — pulled live "
        "from the World Bank's public WDI API. Pick any mix of indicators across domains "
        "below; note WDI has no race/ethnicity variable for these countries — that's a real "
        "gap in the data, not a missing checkbox here."
    )

    _indicator_to_category = {
        code: cat for cat, inds in WORLD_BANK_INDICATOR_CATEGORIES.items() for code in inds
    }

    wb_cols = st.columns([2, 3, 1])
    with wb_cols[0]:
        chosen_countries = st.multiselect(
            "Countries", options=list(SUB_SAHARAN_COUNTRIES.keys()),
            default=["ZW", "GH", "KE"],
            format_func=lambda c: f"{SUB_SAHARAN_COUNTRIES[c]} ({c})",
        )
    with wb_cols[1]:
        chosen_indicators = st.multiselect(
            "Indicators (any mix of domains)",
            options=list(WORLD_BANK_INDICATORS.keys()),
            default=WORLD_BANK_DEFAULT_INDICATORS,
            format_func=lambda i: f"[{_indicator_to_category[i]}] {WORLD_BANK_INDICATORS[i]}",
        )
    with wb_cols[2]:
        st.write("")
        st.write("")
        pull = st.button("Pull live data", use_container_width=True)

    if pull:
        if not chosen_countries or not chosen_indicators:
            st.warning("Pick at least one country and one indicator.")
        else:
            from dat_framework.data.worldbank import fetch_all_indicators, latest_value_per_country

            cache_key = (tuple(chosen_countries), tuple(chosen_indicators))
            if cache_key in st.session_state.get("_wb_cache", {}):
                wb_df = st.session_state["_wb_cache"][cache_key]
            else:
                countries_dict = {k: SUB_SAHARAN_COUNTRIES[k] for k in chosen_countries}
                indicators_dict = {k: WORLD_BANK_INDICATORS[k] for k in chosen_indicators}
                total_calls = len(countries_dict) * len(indicators_dict)
                progress_bar = st.progress(0.0, text=f"Starting... (0/{total_calls} calls)")

                def _wb_progress(done, total, label):
                    progress_bar.progress(done / total, text=f"Fetched: {label} ({done}/{total})")

                try:
                    raw = fetch_all_indicators(
                        countries_dict, indicators_dict, progress_callback=_wb_progress
                    )
                    wb_df = latest_value_per_country(raw)
                    progress_bar.empty()
                    st.session_state.setdefault("_wb_cache", {})[cache_key] = wb_df
                except Exception as exc:  # noqa: BLE001
                    progress_bar.empty()
                    st.error(
                        f"Couldn't reach the World Bank API from this environment ({exc}). "
                        "This needs outbound internet access to api.worldbank.org — check your "
                        "network settings if you're running this somewhere restricted."
                    )
                    wb_df = pd.DataFrame()

            if not wb_df.empty:
                pivot = wb_df.pivot(index="country", columns="indicator_name", values="value")
                st.dataframe(pivot, use_container_width=True)
                fig2 = px.bar(
                    wb_df, x="country", y="value", color="indicator_name", barmode="group",
                    labels={"value": "Latest reported value", "country": "", "indicator_name": ""},
                    color_discrete_sequence=PALETTE["colorway"],
                )
                themed(fig2, height=380)
                st.plotly_chart(fig2, use_container_width=True)
            elif "_wb_cache" not in st.session_state or cache_key not in st.session_state.get("_wb_cache", {}):
                pass  # error already shown above
            else:
                st.warning("World Bank API returned no data for this selection.")

# ---------------------------------------------------------------------------
# TAB 2 — Bias diagnostics
# ---------------------------------------------------------------------------
if section == "diagnostics":
    st.markdown('<div class="dat-section-title">6.0 / 6.1 — Single-attribute fairness metrics</div>', unsafe_allow_html=True)
    st.caption("Comparing the biased baseline model against a classic single-axis fairness fix.")

    baseline_single = single_attribute_metrics(df, y_pred_col="y_pred_baseline")
    corrected_single = single_attribute_metrics(df, y_pred_col="y_pred_single_axis")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Baseline (pre-fairness adjustment)**")
        st.dataframe(
            baseline_single[["DIR", "DPD", "EOD", "DIR_violation", "EOD_violation"]]
            .style.format({"DIR": "{:.2f}", "DPD": "{:.2f}", "EOD": "{:.2f}"}),
            use_container_width=True,
        )
    with c2:
        st.markdown("**After classic single-axis correction**")
        st.dataframe(
            corrected_single[["DIR", "DPD", "EOD", "DIR_violation", "EOD_violation"]]
            .style.format({"DIR": "{:.2f}", "DPD": "{:.2f}", "EOD": "{:.2f}"}),
            use_container_width=True,
        )

    st.info(
        "Notice DIR rises above 0.8 for every attribute after the single-axis fix — "
        "this is the paper's **'mirage of neutrality'**: fairness looks solved, "
        "one attribute at a time.",
    )

    st.divider()
    st.markdown('<div class="dat-section-title">6.2 — Intersectional subgroup metrics</div>', unsafe_allow_html=True)
    st.caption(
        "The same 'corrected' model, now evaluated on intersectional subgroups "
        "(vs. the fully-privileged reference group). Watch DIR fall back below 0.8."
    )

    inter_corrected = intersectional_metrics(
        df, y_pred_col="y_pred_single_axis", min_group_size=min_group_size
    ).sort_values("DIR")

    if inter_corrected.empty:
        st.warning("No intersectional subgroups meet the minimum size threshold — lower it in the sidebar.")
    else:
        n_show = st.slider("Subgroups to show (most disadvantaged first)", 5, min(30, len(inter_corrected)), 10)
        show_df = inter_corrected.head(n_show)
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=show_df.index, y=show_df["DIR"], name="DIR (post single-axis fix)",
            marker_color=[PALETTE["danger"] if v < 0.8 else PALETTE["success"] for v in show_df["DIR"]],
        ))
        fig3.add_hline(y=0.8, line_dash="dash", line_color=PALETTE["muted"],
                        annotation_text="0.80 disparate-impact threshold")
        fig3.update_layout(xaxis_tickangle=-35,
                            yaxis_title="Disparate Impact Ratio", xaxis_title="Intersectional subgroup")
        themed(fig3, height=420)
        st.plotly_chart(fig3, use_container_width=True)
        st.dataframe(
            show_df[["DIR", "DPD", "EOD", "n_marginalized", "DIR_violation"]]
            .style.format({"DIR": "{:.2f}", "DPD": "{:.2f}", "EOD": "{:.2f}"}),
            use_container_width=True,
        )

# ---------------------------------------------------------------------------
# TAB 3 — Statistical validation
# ---------------------------------------------------------------------------
if section == "statistics":
    st.markdown('<div class="dat-section-title">t-tests: single-attribute vs. intersectional</div>', unsafe_allow_html=True)
    baseline_single = single_attribute_metrics(df, y_pred_col="y_pred_baseline")
    baseline_inter = intersectional_metrics(
        df, y_pred_col="y_pred_baseline", min_group_size=min_group_size
    )
    if baseline_inter.empty:
        st.warning("No intersectional subgroups meet the minimum size threshold — lower it in the sidebar.")
    else:
        ttest_df = paired_ttests(baseline_single, baseline_inter)
        st.dataframe(
            ttest_df.style.format({
                "mean_single": "{:.3f}", "mean_intersectional": "{:.3f}",
                "t_statistic": "{:.2f}", "p_value": "{:.4f}",
            }),
            use_container_width=True,
        )
        sig = ttest_df[ttest_df["p_value"] < 0.05]
        if not sig.empty:
            st.success(
                f"{len(sig)}/3 metrics show a statistically significant gap (p<0.05) between "
                "single-attribute and intersectional bias levels.",
            )

    st.divider()
    st.markdown('<div class="dat-section-title">Factorial ANOVA — is intersectional harm non-additive?</div>', unsafe_allow_html=True)
    st.caption(
        "A significant interaction term means two attributes' joint effect on the "
        "prediction isn't just the sum of their individual effects — the mathematical "
        "signature of compounding, non-additive intersectional harm."
    )
    anova_table = factorial_anova(df, outcome_col="y_pred_baseline")
    interaction_rows = anova_table[anova_table.index.str.contains(":")]
    main_rows = anova_table[~anova_table.index.str.contains(":")]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Main effects**")
        st.dataframe(main_rows.style.format({"sum_sq": "{:.2f}", "F": "{:.2f}", "PR(>F)": "{:.2e}"}),
                     use_container_width=True)
    with c2:
        st.markdown("**Pairwise interaction effects**")
        sig_interactions = (interaction_rows["PR(>F)"] < 0.05).sum()
        st.dataframe(interaction_rows.style.format({"sum_sq": "{:.2f}", "F": "{:.2f}", "PR(>F)": "{:.2e}"}),
                     use_container_width=True)
        st.caption(f"{sig_interactions}/{len(interaction_rows)} pairwise interactions significant at p<0.05.")

    st.divider()
    st.markdown('<div class="dat-section-title">Non-additivity spotlight: pick two attributes</div>', unsafe_allow_html=True)
    a1, a2 = st.columns(2)
    with a1:
        attr_a = st.selectbox("Attribute A", PROTECTED_ATTRIBUTES, index=0)
    with a2:
        attr_b = st.selectbox("Attribute B", PROTECTED_ATTRIBUTES, index=1)

    if attr_a != attr_b:
        check = non_additivity_check(df, attr_a, attr_b, outcome_col="y_pred_baseline")
        fig4 = go.Figure()
        labels = ["Neither", f"Only {attr_a}", f"Only {attr_b}", "Both (observed)", "Both (additive prediction)"]
        values = [
            check["reference_rate"], check[f"only_{attr_a}_rate"], check[f"only_{attr_b}_rate"],
            check["observed_joint_rate"], check["additive_prediction"],
        ]
        colors = [PALETTE["muted"], PALETTE["primary"], PALETTE["primary"], PALETTE["danger"], PALETTE["accent"]]
        fig4.add_trace(go.Bar(x=labels, y=values, marker_color=colors))
        fig4.update_layout(yaxis_title="Mean positive prediction rate")
        themed(fig4, height=380)
        st.plotly_chart(fig4, use_container_width=True)
        gap = check["compounding_gap"]
        if check["is_super_additive_harm"]:
            st.warning(
                f"Compounding gap = {gap:.3f}: being **both** {attr_a} and {attr_b} disadvantaged "
                f"hurts more than the sum of each penalty alone — non-additive, super-linear harm.",
            )
        else:
            st.info(f"Compounding gap = {gap:.3f}: no super-additive harm detected for this pair.")
    else:
        st.caption("Pick two different attributes to compare.")

# ---------------------------------------------------------------------------
# TAB 4 — Tier 1-3 DAT / MOO framework
# ---------------------------------------------------------------------------
if section == "moo":
    st.markdown('<div class="dat-section-title">Tier 1 — Sankofa-inspired intersectional mapping</div>', unsafe_allow_html=True)
    st.caption(
        f"{df['subgroup_id'].nunique()} of {2**len(PROTECTED_ATTRIBUTES)} possible intersectional "
        f"subgroups (K = 2^{len(PROTECTED_ATTRIBUTES)}) are populated in this dataset, each assigned "
        "via one-hot encoding of the protected-attribute tuple."
    )

    st.divider()
    st.markdown('<div class="dat-section-title">Tier 2 — Multi-objective optimization sweep</div>', unsafe_allow_html=True)
    st.caption(
        "Runs the paper's 21-point w₀ parameter sweep (0.0 → 1.0, step 0.05) via SLSQP, "
        "trading predictive utility against the composite fairness loss "
        "w1(1-DIR) + w2·EOD + w3·DPD."
    )

    run_col, info_col = st.columns([1, 3])
    with run_col:
        run_moo = st.button("Run MOO sweep", type="primary", use_container_width=True)
    with info_col:
        st.caption(
            f"Weights: DIR={weights.w_dir}, EOD={weights.w_eod}, DPD={weights.w_dpd} · "
            f"min subgroup size={min_group_size} · utility floor={utility_floor_pct}% · "
            f"max SLSQP iterations={slsqp_max_iter} · this takes roughly "
            f"{max(1, n_records // 40)}s–{max(2, n_records // 15)}s depending on dataset size."
        )

    if run_moo:
        progress = st.progress(0.0, text="Starting sweep...")

        def _cb(i, total, sol):
            progress.progress(i / total, text=f"w0={sol.w0:.2f} ({i}/{total}) — "
                                               f"accuracy={sol.accuracy:.2%}, fairness_loss={sol.fairness_loss:.3f}")

        with st.spinner("Running 21-point SLSQP sweep..."):
            solutions = run_w0_sweep(
                df, weights, min_group_size=min_group_size,
                utility_floor=utility_floor_pct / 100, max_iter=slsqp_max_iter,
                progress_callback=_cb,
            )
        progress.empty()
        st.session_state["moo_solutions_df"] = solutions_to_dataframe(solutions)
        st.success("Sweep complete.")

    if "moo_solutions_df" in st.session_state:
        solutions_df = st.session_state["moo_solutions_df"]
        pareto_df = extract_pareto_front(solutions_df)

        st.divider()
        st.markdown('<div class="dat-section-title">Tier 3 — Ubuntu-based Pareto frontier &amp; governance layer</div>', unsafe_allow_html=True)
        st.caption(
            "Per the paper: *'final model selection cannot be designated to automated "
            "computation.'* Explore the frontier below and pick the trade-off you're "
            "comfortable with — a human decision, not an automated one."
        )

        fig5 = go.Figure()
        non_pareto = pareto_df[~pareto_df["is_pareto_optimal"]]
        pareto = pareto_df[pareto_df["is_pareto_optimal"]]
        fig5.add_trace(go.Scatter(
            x=non_pareto["fairness_loss"], y=non_pareto["accuracy_pct"],
            mode="markers", name="Dominated solutions",
            marker=dict(color="#C9C4B4", size=9),
            text=[f"w0={w:.2f}" for w in non_pareto["w0"]],
        ))
        fig5.add_trace(go.Scatter(
            x=pareto["fairness_loss"], y=pareto["accuracy_pct"],
            mode="markers+lines", name="Pareto-optimal front",
            marker=dict(color=PALETTE["accent"], size=12, symbol="diamond"),
            line=dict(color=PALETTE["accent"]),
            text=[f"w0={w:.2f}" for w in pareto["w0"]],
        ))
        fig5.update_layout(
            xaxis_title="Fairness loss (lower = fairer)",
            yaxis_title="Predictive accuracy (%)",
            hovermode="closest",
        )
        themed(fig5, height=460)
        st.plotly_chart(fig5, use_container_width=True)

        st.markdown("#### Walk the frontier yourself")
        w0_choice = st.select_slider(
            "w₀ — utility/fairness trade-off (0 = prioritize fairness, 1 = prioritize utility)",
            options=list(np.round(solutions_df["w0"].to_numpy(), 2)),
            value=float(recommend_solution(pareto_df, priority="balanced")["w0"]),
        )
        chosen = solutions_df[np.isclose(solutions_df["w0"], w0_choice)].iloc[0]
        metric_card_row([
            ("target", "Accuracy", f"{chosen['accuracy_pct']:.1f}%", f"w0={w0_choice:.2f}"),
            ("scale", "Mean DIR", f"{chosen['mean_DIR']:.2f}", "Disparate Impact Ratio"),
            ("shield", "Utility Retention", f"{chosen['utility_retention_pct']:.1f}%", "vs. baseline"),
            ("trend_down", "Fairness Loss", f"{chosen['fairness_loss']:.3f}", "Lower is fairer"),
        ])

        recommended = recommend_solution(pareto_df, utility_floor_pct=float(utility_floor_pct), priority="balanced")
        st.markdown(
            f"""
            <div class="dat-inline-icon-row">
                {icon_html("bulb", size=18, color=PALETTE["accent"])}
                <p style="color:{PALETTE['muted']};font-size:0.875rem;">
                    A balanced starting point (&ge;{utility_floor_pct}% utility retention, lowest fairness loss):
                    <strong>w&#8320; = {recommended['w0']:.2f}</strong> &mdash; accuracy {recommended['accuracy_pct']:.1f}%,
                    mean DIR {recommended['mean_DIR']:.2f}. This is a suggestion to anchor discussion,
                    not the final word &mdash; that&apos;s the governance layer&apos;s job, not the optimizer&apos;s.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Full sweep table"):
            st.dataframe(
                pareto_df.style.format({
                    "mean_DIR": "{:.3f}", "mean_DPD": "{:.3f}", "mean_EOD": "{:.3f}",
                    "fairness_loss": "{:.3f}", "accuracy_pct": "{:.2f}",
                    "utility_retention_pct": "{:.1f}",
                }),
                use_container_width=True,
            )
    else:
        st.info("Click **Run MOO sweep** above to populate the Pareto frontier.")

st.markdown(
    f"""
    <div style="text-align:center; color:{PALETTE['muted']}; font-size:0.78rem; margin-top:2rem;">
        Built for research. Designed for impact. &nbsp;|&nbsp; DAT Ubuntu Fairness © 2026
    </div>
    """,
    unsafe_allow_html=True,
)
