import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from detection.etl import LogLoader
from detection.features import FeatureExtractor
from detection.models import SupervisedAttackClassifier
from detection.rules import RuleEngine
from detection.comparison_eval import run_comparison, evaluate_approach, rules_only_prediction
from detection.alert_exporter import AlertExplainer
from detection.mitre_mapping import get_coverage_matrix, get_mitre_for_attack_type
from detection.response import ResponseEngine
from detection.streaming import StreamingPipeline

# ═══════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ITDR · Security Operations Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════
# DESIGN SYSTEM — Colors, Fonts, Components
# ═══════════════════════════════════════════════════════════
# Palette
BG_PRIMARY = "#0a0e17"
BG_CARD = "#111827"
BG_CARD_HOVER = "#1a2332"
BORDER = "#1e293b"
BORDER_ACCENT = "#0ea5e9"
TEXT_PRIMARY = "#e2e8f0"
TEXT_SECONDARY = "#94a3b8"
TEXT_MUTED = "#64748b"
ACCENT = "#0ea5e9"       # Cyan
ACCENT_2 = "#6366f1"     # Indigo
CRITICAL = "#ef4444"
HIGH = "#f97316"
MEDIUM = "#eab308"
LOW = "#22c55e"
SUCCESS = "#10b981"

# Plotly theme
PLOTLY_LAYOUT = dict(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color=TEXT_SECONDARY, size=12),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor='#1e293b', zerolinecolor='#1e293b'),
    yaxis=dict(gridcolor='#1e293b', zerolinecolor='#1e293b'),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT_SECONDARY)),
    colorway=[ACCENT, ACCENT_2, SUCCESS, HIGH, CRITICAL, MEDIUM, '#8b5cf6', '#ec4899'],
)

def apply_plotly_theme(fig):
    """Apply consistent dark theme to all charts."""
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


# ═══════════════════════════════════════════════════════════
# GLOBAL CSS
# ═══════════════════════════════════════════════════════════
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Reset & Base ────────────────────────────── */
    .stApp {{
        background: {BG_PRIMARY};
        font-family: 'Inter', -apple-system, sans-serif;
    }}
    .stApp header {{
        background: transparent !important;
    }}
    .block-container {{
        padding-top: 1rem !important;
        padding-bottom: 0.5rem !important;
        max-width: 100% !important;
    }}

    /* ── Header Bar ──────────────────────────────── */
    .soc-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 0;
        border-bottom: 1px solid {BORDER};
        margin-bottom: 12px;
    }}
    .soc-header-left {{
        display: flex;
        align-items: center;
        gap: 14px;
    }}
    .soc-logo {{
        width: 36px;
        height: 36px;
        background: linear-gradient(135deg, {ACCENT}, {ACCENT_2});
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
    }}
    .soc-title {{
        font-size: 1.25rem;
        font-weight: 700;
        color: {TEXT_PRIMARY};
        letter-spacing: -0.02em;
    }}
    .soc-subtitle {{
        font-size: 0.78rem;
        color: {TEXT_MUTED};
        margin-top: 1px;
    }}
    .soc-status {{
        display: flex;
        align-items: center;
        gap: 20px;
    }}
    .soc-status-item {{
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 0.78rem;
        color: {TEXT_MUTED};
    }}
    .soc-status-dot {{
        width: 7px;
        height: 7px;
        border-radius: 50%;
        animation: pulse-dot 2s ease-in-out infinite;
    }}
    @keyframes pulse-dot {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.4; }}
    }}

    /* ── Metric Cards ────────────────────────────── */
    .metric-row {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 12px;
        margin-bottom: 20px;
    }}
    .metric-card {{
        background: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 16px 18px;
        transition: border-color 0.2s;
    }}
    .metric-card:hover {{
        border-color: {BORDER_ACCENT};
    }}
    .metric-label {{
        font-size: 0.72rem;
        font-weight: 500;
        color: {TEXT_MUTED};
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }}
    .metric-value {{
        font-size: 1.7rem;
        font-weight: 700;
        color: {TEXT_PRIMARY};
        font-family: 'JetBrains Mono', monospace;
        line-height: 1;
    }}
    .metric-delta {{
        font-size: 0.72rem;
        margin-top: 6px;
        display: flex;
        align-items: center;
        gap: 4px;
    }}
    .metric-delta.positive {{ color: {SUCCESS}; }}
    .metric-delta.negative {{ color: {CRITICAL}; }}
    .metric-delta.neutral {{ color: {TEXT_MUTED}; }}

    /* ── Section Headers ─────────────────────────── */
    .section-header {{
        font-size: 0.78rem;
        font-weight: 600;
        color: {TEXT_SECONDARY};
        text-transform: uppercase;
        letter-spacing: 0.06em;
        padding-bottom: 6px;
        border-bottom: 2px solid {ACCENT}33;
        margin-bottom: 10px;
        margin-top: 4px;
    }}

    /* ── Severity Badges ─────────────────────────── */
    .badge {{
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}
    .badge-critical {{ background: rgba(239,68,68,0.15); color: {CRITICAL}; border: 1px solid rgba(239,68,68,0.3); }}
    .badge-high {{ background: rgba(249,115,22,0.15); color: {HIGH}; border: 1px solid rgba(249,115,22,0.3); }}
    .badge-medium {{ background: rgba(234,179,8,0.15); color: {MEDIUM}; border: 1px solid rgba(234,179,8,0.3); }}
    .badge-low {{ background: rgba(34,197,94,0.15); color: {LOW}; border: 1px solid rgba(34,197,94,0.3); }}

    /* ── Data Panel ───────────────────────────────── */
    .data-panel {{
        background: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 14px;
    }}
    .data-panel-title {{
        font-size: 0.78rem;
        font-weight: 600;
        color: {TEXT_SECONDARY};
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 6px;
    }}

    /* ── Streamlit Overrides ──────────────────────── */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0;
        background: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 4px;
        margin-bottom: 18px;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 6px;
        padding: 8px 20px;
        font-size: 0.82rem;
        font-weight: 500;
        color: {TEXT_MUTED};
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, {ACCENT}22, {ACCENT_2}22) !important;
        color: {ACCENT} !important;
        border: 1px solid {ACCENT}44;
    }}
    .stTabs [data-baseweb="tab-highlight"] {{
        display: none;
    }}
    .stTabs [data-baseweb="tab-border"] {{
        display: none;
    }}
    div[data-testid="stDataFrame"] {{
        border: 1px solid {BORDER};
        border-radius: 8px;
    }}
    /* Removed buggy slider borders */
    div[data-testid="stSelectbox"] > div > div {{
        background: {BG_CARD};
        border-color: {BORDER};
    }}
    div[data-testid="stExpander"] {{
        border: 1px solid {BORDER};
        border-radius: 10px;
        background: {BG_CARD};
    }}
    div[data-testid="stMetric"] {{
        background: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 12px 14px;
        transition: border-color 0.3s, box-shadow 0.3s;
    }}
    div[data-testid="stMetric"]:hover {{
        border-color: {ACCENT}66;
        box-shadow: 0 0 12px {ACCENT}15;
    }}
    div[data-testid="stMetricLabel"] {{
        font-size: 0.68rem !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}
    div[data-testid="stMetricDelta"] {{
        font-size: 0.65rem !important;
    }}
    .stTabs {{
        margin-top: 4px !important;
    }}

    /* ── Architecture Flow ─────────────────────── */
    .arch-flow {{
        display: flex;
        align-items: center;
        justify-content: center;
        flex-wrap: wrap;
        gap: 0;
        padding: 10px 0;
    }}
    .arch-node {{
        background: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 10px 14px;
        text-align: center;
        min-width: 120px;
        transition: border-color 0.2s, transform 0.2s;
    }}
    .arch-node:hover {{
        border-color: {ACCENT};
        transform: translateY(-2px);
    }}
    .arch-node-icon {{
        font-size: 1.2rem;
        margin-bottom: 4px;
    }}
    .arch-node-label {{
        font-size: 0.72rem;
        font-weight: 600;
        color: {TEXT_PRIMARY};
    }}
    .arch-node-desc {{
        font-size: 0.65rem;
        color: {TEXT_MUTED};
        margin-top: 2px;
    }}
    .arch-arrow {{
        color: {ACCENT};
        font-size: 1rem;
        padding: 0 6px;
        opacity: 0.5;
    }}

    /* ── Roadmap ─────────────────────────────────── */
    .roadmap-item {{
        display: flex;
        align-items: flex-start;
        gap: 10px;
        padding: 8px 0;
        border-bottom: 1px solid {BORDER};
    }}
    .roadmap-item:last-child {{ border-bottom: none; }}
    .roadmap-check {{
        color: {SUCCESS};
        font-size: 0.85rem;
        margin-top: 1px;
    }}
    .roadmap-pending {{
        color: {TEXT_MUTED};
        font-size: 0.85rem;
        margin-top: 1px;
    }}
    .roadmap-text {{
        font-size: 0.8rem;
        color: {TEXT_SECONDARY};
    }}

    /* ── Hide Streamlit defaults ──────────────────── */
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    div[data-testid="stToolbar"] {{ display: none; }}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# DATA PIPELINE (cached)
# ═══════════════════════════════════════════════════════════
def _paper_comparison_metrics():
    """Capstone-paper summary metrics used for demo mode and README figures."""
    return [
        {
            "approach": "Rules-Only",
            "accuracy": 0.536,
            "precision": 0.148,
            "recall": 0.765,
            "f1_score": 0.248,
            "fpr": 0.499,
        },
        {
            "approach": "ML Behavioral",
            "accuracy": 0.954,
            "precision": 0.735,
            "recall": 0.857,
            "f1_score": 0.791,
            "fpr": 0.035,
        },
        {
            "approach": "Combined",
            "accuracy": 0.525,
            "precision": 0.178,
            "recall": 0.946,
            "f1_score": 0.299,
            "fpr": 0.526,
        },
    ]


def _build_demo_pipeline(critical_t=70, high_t=50):
    """
    Build a deterministic demo dataset so the dashboard works in a fresh clone.

    The real end-to-end path still runs when rba-dataset.csv and saved model
    artifacts are present. This fallback uses the capstone-paper confusion
    counts and is clearly labeled in the UI.
    """
    rng = np.random.default_rng(584)

    # Counts chosen to round to 95.4% accuracy, 73.5% precision, and 3.5% FPR.
    tp_count, fp_count, fn_count, tn_count = 4362, 1573, 729, 43336
    y_true = np.concatenate([
        np.ones(tp_count, dtype=int),
        np.zeros(fp_count, dtype=int),
        np.ones(fn_count, dtype=int),
        np.zeros(tn_count, dtype=int),
    ])
    y_pred = np.concatenate([
        np.ones(tp_count, dtype=int),
        np.ones(fp_count, dtype=int),
        np.zeros(fn_count, dtype=int),
        np.zeros(tn_count, dtype=int),
    ])

    order = rng.permutation(len(y_true))
    y_true = y_true[order]
    y_pred = y_pred[order]

    timestamps = pd.date_range("2020-02-05 00:00:00", periods=len(y_true), freq="2s")
    users = np.array([f"user{idx:04d}@example.com" for idx in rng.integers(1, 1400, len(y_true))])
    countries = rng.choice(
        ["US", "NO", "PL", "BR", "GB", "DE", "IN", "IE"],
        size=len(y_true),
        p=[0.34, 0.22, 0.12, 0.09, 0.08, 0.06, 0.05, 0.04],
    )
    ips = np.array([
        f"203.0.{a}.{b}"
        for a, b in zip(rng.integers(1, 255, len(y_true)), rng.integers(1, 255, len(y_true)))
    ])

    attack_labels = rng.choice(
        ["Password Spray", "Impossible Travel", "Token Theft", "Privilege Escalation", "Suspicious IP"],
        size=len(y_true),
        p=[0.36, 0.24, 0.18, 0.12, 0.10],
    )
    attack_type = np.where(y_true == 1, attack_labels, "Normal")

    probabilities = np.zeros(len(y_true), dtype=float)
    probabilities[(y_true == 1) & (y_pred == 1)] = rng.uniform(
        0.72, 0.99, ((y_true == 1) & (y_pred == 1)).sum()
    )
    probabilities[(y_true == 0) & (y_pred == 1)] = rng.uniform(
        0.70, 0.91, ((y_true == 0) & (y_pred == 1)).sum()
    )
    probabilities[(y_true == 1) & (y_pred == 0)] = rng.uniform(
        0.31, 0.49, ((y_true == 1) & (y_pred == 0)).sum()
    )
    probabilities[(y_true == 0) & (y_pred == 0)] = rng.uniform(
        0.01, 0.29, ((y_true == 0) & (y_pred == 0)).sum()
    )

    df = pd.DataFrame({
        "timestamp": timestamps,
        "upn": users,
        "ip": ips,
        "country": countries,
        "eventType": np.where(y_true == 1, "UserLoggedIn", "UserLogin"),
        "status": np.where((y_true == 1) & (rng.random(len(y_true)) < 0.38), "Failure", "Success"),
        "appName": rng.choice(["Office 365", "AWS Console", "Okta", "Azure Portal"], size=len(y_true)),
        "browser": rng.choice(["Chrome", "Edge", "Firefox", "Safari"], size=len(y_true)),
        "os": rng.choice(["Windows", "macOS", "Linux", "iOS", "Android"], size=len(y_true)),
        "asn": rng.integers(1000, 65000, len(y_true)),
        "is_attack": y_true.astype(bool),
        "attack_type": attack_type,
        "predicted_attack": y_pred,
        "attack_probability": probabilities,
        "final_risk_score": (probabilities * 100).round(1),
        "ml_risk": (probabilities * 100).round(1),
        "rule_risk": 0,
        "rule_details": "",
    })

    df["risk_level"] = df["final_risk_score"].apply(
        lambda x: "Critical" if x >= critical_t else ("High" if x >= high_t else ("Medium" if x >= 30 else "Low"))
    )
    df["mitre_id"] = df["attack_type"].apply(
        lambda x: (get_mitre_for_attack_type(str(x)) or {}).get("id", "") if x != "Normal" else ""
    )
    df["top_signals"] = np.where(
        df["predicted_attack"].astype(bool),
        "ip_attack_rate, user_fail_rate, hour_deviation",
        "No high-risk signal",
    )
    df["rationale"] = np.where(
        df["predicted_attack"].astype(bool),
        "ML ensemble flagged anomalous identity behavior.",
        "Below detection threshold.",
    )
    df.attrs["demo_mode"] = True

    feature_importance = pd.DataFrame({
        "Feature": [
            "ip_attack_rate", "asn_attack_rate", "country_attack_rate", "user_fail_rate",
            "hour_deviation", "ip_sharing_score", "is_failure", "is_new_device",
            "country_freq", "browser_freq",
        ],
        "Importance": [0.19, 0.14, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.05, 0.05],
    })

    rule_alerts = []
    rules = ["Password Spray", "Impossible Travel", "Token Theft", "Privilege Escalation", "Suspicious IP"] * 4
    for idx, rule in enumerate(rules):
        entity_type = "ip" if rule in ("Password Spray", "Suspicious IP") else "user"
        entity = str(df.iloc[idx]["ip"] if entity_type == "ip" else df.iloc[idx]["upn"])
        rule_alerts.append({
            "rule": rule,
            "severity": "Critical" if idx % 3 == 0 else "High",
            "entity": entity,
            "entity_type": entity_type,
            "mitre_id": (get_mitre_for_attack_type(rule) or {}).get("id", ""),
        })

    metrics = {
        "accuracy": 0.954,
        "feature_importance": feature_importance,
        "comparison": _paper_comparison_metrics(),
        "attack_probabilities": probabilities,
        "y_true": y_true,
        "dataset_mode": "demo",
    }

    return df, metrics, rule_alerts


@st.cache_data(show_spinner="Initializing detection pipeline...")
def run_pipeline(critical_t=70, high_t=50):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(script_dir, "..", "rba-dataset.csv")

    loader = LogLoader(log_path)
    try:
        df = loader.load_to_dataframe(nrows=200000)
    except FileNotFoundError:
        return _build_demo_pipeline(critical_t, high_t)

    metadata_path = os.path.join(script_dir, "..", "saved_models", "split_metadata.json")
    try:
        with open(metadata_path) as f:
            split_meta = json.load(f)
        split_date = pd.Timestamp(split_meta['split_date'])
        df = df[df['timestamp'] >= split_date].copy()
        if df.empty:
            return _build_demo_pipeline(critical_t, high_t)
        if len(df) > 50000:
            df = df.head(50000)
    except FileNotFoundError:
        df = df.tail(50000).copy()

    rule_engine = RuleEngine(df)
    rule_alerts = rule_engine.run_all()

    extractor_path = os.path.join(script_dir, "..", "saved_models", "feature_extractor.pkl")
    model_path = os.path.join(script_dir, "..", "saved_models", "rba_trained_model.pkl")

    import joblib
    try:
        extractor = joblib.load(extractor_path)
        classifier = SupervisedAttackClassifier()
        classifier.load_model(model_path)
    except FileNotFoundError:
        return _build_demo_pipeline(critical_t, high_t)

    X = extractor.transform(df)
    supervised_results = classifier.predict(X)

    y_true = df['is_attack'].fillna(False).astype(int)
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_true, supervised_results['supervised_pred'].astype(int))

    rf_model = classifier.model.named_estimators_['rf']
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    full_df = df.copy()
    full_df['attack_probability'] = supervised_results['attack_probability'].values
    full_df['predicted_attack'] = supervised_results['supervised_pred'].values
    full_df['final_risk_score'] = (full_df['attack_probability'] * 100).round(1)

    full_df['risk_level'] = full_df['final_risk_score'].apply(
        lambda x: 'Critical' if x >= critical_t else ('High' if x >= high_t else ('Medium' if x >= 30 else 'Low'))
    )
    full_df['ml_risk'] = full_df['final_risk_score']
    full_df['rule_risk'] = 0
    full_df['rule_details'] = ''

    feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
    explainer = AlertExplainer(feature_importances, X.mean(), X.std())
    flagged_mask = full_df['predicted_attack'].astype(bool)
    explanations = explainer.explain_batch(X, flagged_mask, top_n=5)
    full_df['top_signals'] = explanations['top_signals']
    full_df['rationale'] = explanations['rationale']

    full_df['mitre_id'] = full_df['attack_type'].apply(
        lambda x: (get_mitre_for_attack_type(str(x)) or {}).get('id', '') if pd.notna(x) else ''
    )

    ml_pred = pd.Series(supervised_results['supervised_pred'].values, index=y_true.index).astype(int)
    y_rules = rules_only_prediction(df, rule_alerts)
    rules_metrics = evaluate_approach(y_true, y_rules, "Rules-Only")
    ml_metrics = evaluate_approach(y_true, ml_pred, "ML Behavioral")
    y_combined = ((y_rules == 1) | (ml_pred == 1)).astype(int)
    combined_metrics = evaluate_approach(y_true, y_combined, "Combined")

    metrics = {
        'accuracy': accuracy,
        'feature_importance': feature_importance,
        'comparison': [rules_metrics, ml_metrics, combined_metrics],
        'attack_probabilities': supervised_results['attack_probability'].values,
        'y_true': y_true.values,
    }

    return full_df, metrics, rule_alerts


# ═══════════════════════════════════════════════════════════
# HEADER BAR
# ═══════════════════════════════════════════════════════════
script_dir = os.path.dirname(os.path.abspath(__file__))
metadata_path = os.path.join(script_dir, "..", "saved_models", "split_metadata.json")
split_meta = {}
try:
    with open(metadata_path) as f:
        split_meta = json.load(f)
except FileNotFoundError:
    pass

st.markdown(f"""
<div class="soc-header">
    <div class="soc-header-left">
        <div class="soc-logo">🛡️</div>
        <div>
            <div class="soc-title">ITDR Security Operations Center</div>
            <div class="soc-subtitle">Identity Threat Detection & Response · IST 584 Capstone</div>
        </div>
    </div>
    <div class="soc-status">
        <div class="soc-status-item">
            <div class="soc-status-dot" style="background:{SUCCESS};"></div>
            ML Engine Online
        </div>
        <div class="soc-status-item">
            <div class="soc-status-dot" style="background:{SUCCESS};"></div>
            Rule Engine Active
        </div>
        <div class="soc-status-item">
            <div class="soc-status-dot" style="background:{ACCENT};"></div>
            v2.0 · Test Data Feb 5-6
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# SIDEBAR — Controls, What's New, Streaming
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding: 10px 0 16px 0;">
        <div style="font-size:1.8rem;">🛡️</div>
        <div style="font-size:0.95rem; font-weight:700; color:{TEXT_PRIMARY};">ITDR v2.0</div>
        <div style="font-size:0.68rem; color:{TEXT_MUTED};">Security Operations Center</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Live Streaming Mode
    st.markdown(f"""<div style="font-size:0.78rem; font-weight:600; color:{TEXT_SECONDARY};
        text-transform:uppercase; letter-spacing:0.05em; margin-bottom:8px;">⚡ Live Streaming Mode</div>""", unsafe_allow_html=True)
    streaming_enabled = st.toggle("Enable Live Simulation", value=False, key="streaming_toggle")

    if streaming_enabled:
        batch_size = st.slider("Batch Size", 10, 500, 100, step=10, key="stream_batch")
        st.caption("Processes events in real-time batches through the full ML pipeline.")

    st.divider()

    # Detection Configuration
    st.markdown(f"""<div style="font-size:0.78rem; font-weight:600; color:{TEXT_SECONDARY};
        text-transform:uppercase; letter-spacing:0.05em; margin-bottom:8px;">🎛️ Detection Config</div>""", unsafe_allow_html=True)
    critical_thresh = st.slider("Critical Threshold", 50, 100, 70, key="thresh_critical")
    high_thresh = st.slider("High Threshold", 20, critical_thresh, min(50, critical_thresh), key="thresh_high")
    st.caption(f"Events ≥ {critical_thresh} → Critical, ≥ {high_thresh} → High")

    st.divider()

    # What's New
    st.markdown(f"""<div style="font-size:0.78rem; font-weight:600; color:{TEXT_SECONDARY};
        text-transform:uppercase; letter-spacing:0.05em; margin-bottom:8px;">🆕 What's New (v2.0)</div>""", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:0.72rem; color:{TEXT_MUTED}; line-height:1.7;">
        ✅ Ensemble ML (RF + HistGBT)<br>
        ✅ Temporal train/test split<br>
        ✅ MITRE ATT&CK mapping<br>
        ✅ Automated response engine<br>
        ✅ Per-alert ML explainability<br>
        ✅ Live streaming simulation<br>
        ✅ Precision-recall tuning<br>
        ✅ SOC dashboard overhaul
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Data Info
    st.markdown(f"""<div style="font-size:0.78rem; font-weight:600; color:{TEXT_SECONDARY};
        text-transform:uppercase; letter-spacing:0.05em; margin-bottom:8px;">📂 Data Info</div>""", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:0.72rem; color:{TEXT_MUTED}; line-height:1.7;">
        Source: RBA Dataset (9GB)<br>
        Train: Feb 3-4 (121K events)<br>
        Test: Feb 5-6 (50K events)<br>
        Features: 24 behavioral<br>
        Rules: 7 detections
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════
result = run_pipeline(critical_thresh, high_thresh)
if result is None or result[0] is None:
    st.error("Pipeline failed. Ensure `rba-dataset.csv` and trained models exist.")
    st.stop()

df, metrics, rule_alerts = result

if metrics.get("dataset_mode") == "demo":
    st.info(
        "Demo mode: committed RBA dataset/model artifacts are not included, so this view uses "
        "deterministic capstone-summary data. Add rba-dataset.csv and saved_models/ to run "
        "the full training/evaluation pipeline."
    )

# Pre-compute common values
total_events = len(df)
y_true = df['is_attack'].astype(int)
y_pred = df['predicted_attack'].astype(int)
tp = int(((y_true == 1) & (y_pred == 1)).sum())
tn = int(((y_true == 0) & (y_pred == 0)).sum())
fp = int(((y_true == 0) & (y_pred == 1)).sum())
fn = int(((y_true == 1) & (y_pred == 0)).sum())
precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_val = 2 * precision_val * recall_val / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0
accuracy_val = (tp + tn) / total_events

high_risk = df[df['final_risk_score'] >= high_thresh]
critical_count = len(df[df['final_risk_score'] >= critical_thresh])
high_count = len(df[(df['final_risk_score'] >= high_thresh) & (df['final_risk_score'] < critical_thresh)])
compromised_users = high_risk['upn'].nunique()


# ═══════════════════════════════════════════════════════════
# TOP METRICS ROW — Two rows of 4 to prevent truncation
# ═══════════════════════════════════════════════════════════
c1, c2, c3, c4 = st.columns(4)
c1.metric("Events Analyzed", f"{total_events:,}", "Held-out test set", delta_color="off")
c2.metric("Critical Alerts", f"{critical_count:,}", f"{critical_count/total_events*100:.1f}% of events", delta_color="off")
c3.metric("High Risk", f"{high_count:,}", f"Score 50-69" if high_count > 0 else "None detected", delta_color="off")
c4.metric("Users Impacted", f"{compromised_users:,}", f"{compromised_users} unique identities", delta_color="off")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Precision", f"{precision_val:.1%}", f"TP: {tp:,} / FP: {fp:,}", delta_color="off")
c6.metric("Recall", f"{recall_val:.1%}", f"Caught {tp:,} of {tp+fn:,}", delta_color="off")
c7.metric("F1 Score", f"{f1_val:.1%}", "Precision-Recall balance", delta_color="off")
c8.metric("Accuracy", f"{accuracy_val:.1%}", "Ensemble RF+HistGBT", delta_color="off")


# ═══════════════════════════════════════════════════════════
# TAB NAVIGATION
# ═══════════════════════════════════════════════════════════
tab_overview, tab_alerts, tab_detection, tab_model, tab_research = st.tabs([
    "📊 Overview",
    "🚨 Alerts & Response",
    "🎯 Detection Analytics",
    "🤖 Model Performance",
    "🔬 Research"
])


# ─────────────────────────────────────────────────────────
# TAB 1: OVERVIEW
# ─────────────────────────────────────────────────────────
with tab_overview:
    col_main, col_side = st.columns([5, 2])

    with col_main:
        # Threat Timeline — stacked area showing attack vs normal traffic
        st.markdown('<div class="section-header">⏱ Threat Activity Timeline</div>', unsafe_allow_html=True)
        df_time = df.copy()
        df_time['time_bin'] = df_time['timestamp'].dt.floor('30min')
        df_time['category'] = np.where(df_time['is_attack'] == True, 'Attacks Detected', 'Normal Traffic')
        timeline_agg = df_time.groupby(['time_bin', 'category']).size().reset_index(name='count')

        fig_tl = go.Figure()
        # Normal traffic — subtle background
        normal = timeline_agg[timeline_agg['category'] == 'Normal Traffic']
        attacks = timeline_agg[timeline_agg['category'] == 'Attacks Detected']
        fig_tl.add_trace(go.Scatter(
            x=normal['time_bin'], y=normal['count'], name='Normal Traffic',
            fill='tozeroy', fillcolor='rgba(14,165,233,0.08)',
            line=dict(color='rgba(14,165,233,0.4)', width=1),
            hovertemplate='%{y:,} events<extra>Normal</extra>',
        ))
        fig_tl.add_trace(go.Scatter(
            x=attacks['time_bin'], y=attacks['count'], name='Threats Detected',
            fill='tozeroy', fillcolor='rgba(239,68,68,0.25)',
            line=dict(color=CRITICAL, width=2),
            hovertemplate='%{y:,} attacks<extra>Threats</extra>',
        ))
        apply_plotly_theme(fig_tl)
        fig_tl.update_layout(
            height=280, xaxis_title="", yaxis_title="Events / 30 min",
            legend=dict(orientation='h', y=1.08, x=0),
            hovermode='x unified',
        )
        st.plotly_chart(fig_tl, use_container_width=True)

    with col_side:
        # Detection Outcome — donut showing ML performance story
        st.markdown('<div class="section-header">🎯 Detection Outcome</div>', unsafe_allow_html=True)
        fig_donut = go.Figure(data=[go.Pie(
            labels=['True Positives', 'True Negatives', 'False Positives', 'False Negatives'],
            values=[tp, tn, fp, fn],
            hole=0.6,
            marker=dict(
                colors=[SUCCESS, ACCENT, HIGH, CRITICAL],
                line=dict(color=BG_PRIMARY, width=2),
            ),
            textinfo='label+percent',
            textfont=dict(size=10, color=TEXT_PRIMARY),
            hovertemplate='%{label}: %{value:,}<extra></extra>',
            sort=False,
        )])
        apply_plotly_theme(fig_donut)
        fig_donut.update_layout(
            height=280, showlegend=False,
            annotations=[dict(
                text=f'<b>{accuracy_val:.1%}</b><br>Accuracy',
                x=0.5, y=0.5, font=dict(size=16, color=TEXT_PRIMARY), showarrow=False,
            )],
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # Bottom row
    col_hourly, col_geo, col_attacks = st.columns([1, 1, 1])

    with col_hourly:
        st.markdown('<div class="section-header">📈 Hourly Activity Pattern</div>', unsafe_allow_html=True)
        df_hour = df.copy()
        df_hour['hour'] = df_hour['timestamp'].dt.hour
        hourly = df_hour.groupby('hour').agg(
            total=('hour', 'count'),
            attacks=('is_attack', 'sum')
        ).reset_index()

        fig_hr = go.Figure()
        fig_hr.add_trace(go.Scatter(
            x=hourly['hour'], y=hourly['total'], name='All Events',
            fill='tozeroy', fillcolor=f'rgba(14,165,233,0.15)',
            line=dict(color=ACCENT, width=2),
        ))
        fig_hr.add_trace(go.Scatter(
            x=hourly['hour'], y=hourly['attacks'], name='Attacks',
            fill='tozeroy', fillcolor=f'rgba(239,68,68,0.2)',
            line=dict(color=CRITICAL, width=2),
        ))
        apply_plotly_theme(fig_hr)
        fig_hr.update_layout(height=260, xaxis_title="Hour (UTC)", yaxis_title="",
                             legend=dict(orientation='h', y=1.15, x=0),
                             hovermode='x unified')
        st.plotly_chart(fig_hr, use_container_width=True)

    with col_geo:
        st.markdown('<div class="section-header">🌍 Top Access Countries</div>', unsafe_allow_html=True)
        loc_counts = df.groupby('country').size().reset_index(name='count').sort_values('count', ascending=True).tail(8)
        fig_geo = px.bar(loc_counts, x='count', y='country', orientation='h',
                         color='count', color_continuous_scale=[[0, ACCENT_2], [1, ACCENT]])
        apply_plotly_theme(fig_geo)
        fig_geo.update_layout(height=260, xaxis_title="", yaxis_title="", coloraxis_showscale=False)
        fig_geo.update_traces(marker_line_width=0)
        st.plotly_chart(fig_geo, use_container_width=True)

    with col_attacks:
        st.markdown('<div class="section-header">⚔️ Attack Detection by Type</div>', unsafe_allow_html=True)
        atk_df = df[df['is_attack'] == True]
        if not atk_df.empty:
            atk_counts = atk_df.groupby('attack_type').size().reset_index(name='count')
            detected_counts = df[(df['is_attack'] == True) & (df['predicted_attack'].astype(bool))].groupby('attack_type').size()
            atk_counts['detected'] = atk_counts['attack_type'].map(detected_counts).fillna(0).astype(int)
            atk_counts = atk_counts.sort_values('count', ascending=True)

            fig_atk = go.Figure()
            fig_atk.add_trace(go.Bar(y=atk_counts['attack_type'], x=atk_counts['count'],
                                     name='Total', orientation='h', marker_color=f'rgba(14,165,233,0.3)',
                                     marker_line_width=0))
            fig_atk.add_trace(go.Bar(y=atk_counts['attack_type'], x=atk_counts['detected'],
                                     name='Detected', orientation='h', marker_color=SUCCESS,
                                     marker_line_width=0))
            apply_plotly_theme(fig_atk)
            fig_atk.update_layout(height=260, barmode='overlay', xaxis_title="", yaxis_title="",
                                  legend=dict(orientation='h', y=1.15, x=0))
            st.plotly_chart(fig_atk, use_container_width=True)
        else:
            st.info("No attacks in dataset.")

    # ── Live Streaming Panel ──────────────────────────────
    if streaming_enabled:
        st.markdown('<div class="section-header">⚡ Live Streaming Simulation</div>', unsafe_allow_html=True)

        stream_col1, stream_col2 = st.columns([3, 1])

        with stream_col1:
            st.markdown(f"""
            <div class="data-panel">
                <div class="data-panel-title">🔄 Real-Time Event Processing</div>
                <div style="color:{TEXT_MUTED}; font-size:0.8rem;">
                    This mode processes events in batches through the full ML pipeline
                    (ETL → Feature Extraction → ML Ensemble → Risk Scoring) to simulate
                    live SOC operation. Each batch runs through the identical pipeline used
                    in the static analysis above.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with stream_col2:
            run_stream = st.button("▶ Process Next Batch", type="primary", use_container_width=True)
            auto_stream = st.toggle("🔁 Auto-Stream (Continuous)")

        if run_stream or auto_stream:
            try:
                csv_path = os.path.join(script_dir, "..", "rba-dataset.csv")
                model_path = os.path.join(script_dir, "..", "saved_models", "rba_trained_model.pkl")
                extractor_path = os.path.join(script_dir, "..", "saved_models", "feature_extractor.pkl")

                if 'stream_pipeline' not in st.session_state:
                    st.session_state.stream_pipeline = StreamingPipeline(
                        csv_path, model_path, extractor_path,
                        batch_size=batch_size
                    )

                pipeline = st.session_state.stream_pipeline
                pipeline.batch_size = batch_size
                with st.spinner("Processing batch through ML pipeline..."):
                    batch_df = pipeline.process_next_batch()

                if batch_df is not None:
                    stats = pipeline.get_stats()
                    s1, s2, s3, s4 = st.columns(4)
                    s1.metric("Total Processed", f"{stats['total_processed']:,}")
                    s2.metric("Alerts Triggered", f"{stats['total_alerts']:,}")
                    s3.metric("Events/sec", f"{stats['events_per_second']:.0f}")
                    s4.metric("Batch Size", f"{len(batch_df):,}")

                    st.markdown('<div class="section-header">📡 Latest Scored Events</div>', unsafe_allow_html=True)
                    latest = pipeline.get_latest_events(30)
                    display_cols = ['timestamp', 'upn', 'final_risk_score', 'risk_level', 'ip', 'country']
                    available = [c for c in display_cols if c in latest.columns]
                    st.dataframe(
                        latest[available],
                        use_container_width=True, hide_index=True, height=250,
                        column_config={
                            "final_risk_score": st.column_config.ProgressColumn(
                                "Risk Score", min_value=0, max_value=100, format="%d"
                            ),
                        }
                    )
                    if auto_stream:
                        import time
                        time.sleep(1.0)
                        st.rerun()
                else:
                    st.success("✅ End of dataset reached. Reset to start over.")
            except Exception as e:
                st.error(f"Streaming error: {e}")

# ─────────────────────────────────────────────────────────
# TAB 2: ALERTS & RESPONSE
# ─────────────────────────────────────────────────────────
with tab_alerts:
    # Styled filter bar
    fc1, fc2, fc3 = st.columns([1, 1, 2])
    with fc1:
        min_score = st.slider("🔍 Minimum Risk Score", 0, 100, 50, key="alert_filter")
    with fc2:
        severity_filter = st.selectbox("Severity", ["All", "Critical", "High", "Medium"], key="sev_filter")

    alert_df = df[df['final_risk_score'] >= min_score].sort_values('final_risk_score', ascending=False)
    if severity_filter != "All":
        alert_df = alert_df[alert_df['risk_level'] == severity_filter]

    # Alert summary strip
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Total Alerts", f"{len(alert_df):,}")
    a2.metric("Critical", f"{len(alert_df[alert_df['risk_level']=='Critical']):,}")
    a3.metric("Rule Alerts", f"{len(rule_alerts)}")
    a4.metric("Unique Users", f"{alert_df['upn'].nunique()}")

    col_table, col_response = st.columns([3, 2])

    with col_table:
        st.markdown('<div class="section-header">🚨 Priority Alerts</div>', unsafe_allow_html=True)
        display_cols = ['timestamp', 'upn', 'final_risk_score', 'risk_level', 'mitre_id', 'rationale', 'ip', 'country']
        available = [c for c in display_cols if c in alert_df.columns]
        st.dataframe(
            alert_df[available].head(200),
            use_container_width=True,
            hide_index=True,
            height=500,
            column_config={
                "final_risk_score": st.column_config.ProgressColumn(
                    "Risk Score", min_value=0, max_value=100, format="%d"
                ),
                "timestamp": st.column_config.DatetimeColumn("Time", format="MMM DD, HH:mm"),
            }
        )

    with col_response:
        st.markdown('<div class="section-header">⚡ Automated Response Actions</div>', unsafe_allow_html=True)
        if rule_alerts:
            response_engine = ResponseEngine()
            response_actions = response_engine.process_alerts(rule_alerts)
            summary = response_engine.get_response_summary()

            r1, r2 = st.columns(2)
            r1.metric("Actions Taken", summary['total_actions'])
            r2.metric("Simulated", summary['by_status'].get('Simulated', 0))

            if response_actions:
                actions_data = [a.to_dict() for a in response_actions[:80]]
                actions_df = pd.DataFrame(actions_data)
                st.dataframe(
                    actions_df[['action', 'target', 'severity', 'status']],
                    use_container_width=True,
                    hide_index=True,
                    height=360,
                )
        else:
            st.info("No rule-based alerts triggered.")


# ─────────────────────────────────────────────────────────
# TAB 3: DETECTION ANALYTICS
# ─────────────────────────────────────────────────────────
with tab_detection:
    col_mitre, col_attacks = st.columns([3, 2])

    with col_mitre:
        st.markdown('<div class="section-header">🎯 MITRE ATT&CK Coverage Matrix</div>', unsafe_allow_html=True)
        coverage = get_coverage_matrix()
        coverage_df = pd.DataFrame(coverage)
        st.dataframe(
            coverage_df[['Technique ID', 'Technique Name', 'Tactic', 'Detection Rule', 'Severity']],
            use_container_width=True, hide_index=True, height=380,
        )

    with col_attacks:
        st.markdown('<div class="section-header">📊 Detection by Attack Type</div>', unsafe_allow_html=True)
        atk = df[df['is_attack'] == True]
        if not atk.empty:
            attack_types = atk.groupby('attack_type').agg(
                count=('is_attack', 'count'),
                avg_score=('final_risk_score', 'mean')
            ).reset_index()
            detected = df[(df['is_attack'] == True) & (df['predicted_attack'].astype(bool))].groupby('attack_type').size()
            attack_types['detected'] = attack_types['attack_type'].map(detected).fillna(0).astype(int)
            attack_types['rate'] = (attack_types['detected'] / attack_types['count'] * 100).round(1)
            attack_types['MITRE'] = attack_types['attack_type'].apply(
                lambda x: (get_mitre_for_attack_type(str(x)) or {}).get('id', '')
            )
            attack_types.columns = ['Attack Type', 'Total', 'Avg Score', 'Detected', 'Rate %', 'MITRE']
            st.dataframe(attack_types, use_container_width=True, hide_index=True, height=180)

        # Tactic coverage pie
        tactic_counts = coverage_df['Tactic'].value_counts().reset_index()
        tactic_counts.columns = ['Tactic', 'Rules']
        fig_tac = px.pie(tactic_counts, values='Rules', names='Tactic', hole=0.5,
                         color_discrete_sequence=[ACCENT, ACCENT_2, SUCCESS, HIGH, MEDIUM])
        apply_plotly_theme(fig_tac)
        fig_tac.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_tac, use_container_width=True)

    # Architecture flow
    with st.expander("System Architecture"):
        st.markdown(f"""
        <div class="arch-flow">
            <div class="arch-node">
                <div class="arch-node-icon">📥</div>
                <div class="arch-node-label">ETL Pipeline</div>
                <div class="arch-node-desc">9GB RBA Dataset</div>
            </div>
            <div class="arch-arrow">→</div>
            <div class="arch-node">
                <div class="arch-node-icon">🔬</div>
                <div class="arch-node-label">Feature Engine</div>
                <div class="arch-node-desc">24 Features</div>
            </div>
            <div class="arch-arrow">→</div>
            <div class="arch-node" style="border-color:{ACCENT};">
                <div class="arch-node-icon">🤖</div>
                <div class="arch-node-label">ML Ensemble</div>
                <div class="arch-node-desc">RF + HistGBT</div>
            </div>
            <div class="arch-arrow">→</div>
            <div class="arch-node">
                <div class="arch-node-icon">📊</div>
                <div class="arch-node-label">Risk Scoring</div>
                <div class="arch-node-desc">0-100 Unified</div>
            </div>
            <div class="arch-arrow">→</div>
            <div class="arch-node" style="border-color:{CRITICAL};">
                <div class="arch-node-icon">🚨</div>
                <div class="arch-node-label">Alert Console</div>
                <div class="arch-node-desc">MITRE Enriched</div>
            </div>
            <div class="arch-arrow">→</div>
            <div class="arch-node" style="border-color:{SUCCESS};">
                <div class="arch-node-icon">🛡️</div>
                <div class="arch-node-label">Auto-Response</div>
                <div class="arch-node-desc">Playbook Actions</div>
            </div>
        </div>
        <div class="arch-flow" style="margin-top: 8px;">
            <div class="arch-node">
                <div class="arch-node-icon">📜</div>
                <div class="arch-node-label">Rule Engine</div>
                <div class="arch-node-desc">7 Detection Rules</div>
            </div>
            <div class="arch-arrow">→</div>
            <div class="arch-node">
                <div class="arch-node-icon">🎯</div>
                <div class="arch-node-label">MITRE Mapping</div>
                <div class="arch-node-desc">ATT&CK Taxonomy</div>
            </div>
            <div class="arch-arrow">→</div>
            <div class="arch-node">
                <div class="arch-node-icon">💡</div>
                <div class="arch-node-label">Explainability</div>
                <div class="arch-node-desc">Per-Alert Signals</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# TAB 4: MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────
with tab_model:
    col_cm, col_stats, col_feat = st.columns([1, 1, 1])

    with col_cm:
        st.markdown('<div class="section-header">🔢 Confusion Matrix</div>', unsafe_allow_html=True)
        cm_data = [[tn, fp], [fn, tp]]
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm_data,
            x=['Pred Normal', 'Pred Attack'],
            y=['Actual Normal', 'Actual Attack'],
            text=[[f'TN\n{tn:,}', f'FP\n{fp:,}'], [f'FN\n{fn:,}', f'TP\n{tp:,}']],
            texttemplate='%{text}', textfont=dict(size=14, color=TEXT_PRIMARY),
            colorscale=[[0, '#0f172a'], [0.5, '#1e3a5f'], [1, ACCENT]],
            showscale=False, hoverinfo='skip',
        ))
        apply_plotly_theme(fig_cm)
        fig_cm.update_layout(height=320, xaxis_title="", yaxis_title="")
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_stats:
        st.markdown('<div class="section-header">📋 Detection Breakdown</div>', unsafe_allow_html=True)
        stats_metrics = pd.DataFrame({
            'Category': ['True Positives', 'True Negatives', 'False Positives', 'False Negatives'],
            'Count': [tp, tn, fp, fn]
        })
        colors = [SUCCESS, ACCENT, HIGH, CRITICAL]
        fig_bar = px.bar(stats_metrics, x='Count', y='Category', orientation='h',
                         color='Category', color_discrete_sequence=colors)
        apply_plotly_theme(fig_bar)
        fig_bar.update_layout(height=320, showlegend=False, xaxis_title="", yaxis_title="")
        fig_bar.update_traces(marker_line_width=0)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_feat:
        st.markdown('<div class="section-header">🧠 Top Features</div>', unsafe_allow_html=True)
        if metrics and 'feature_importance' in metrics:
            imp = metrics['feature_importance'].head(10).sort_values('Importance')
            fig_fi = px.bar(imp, x='Importance', y='Feature', orientation='h',
                            color='Importance', color_continuous_scale=[[0, ACCENT_2], [1, ACCENT]])
            apply_plotly_theme(fig_fi)
            fig_fi.update_layout(height=320, coloraxis_showscale=False, xaxis_title="", yaxis_title="")
            fig_fi.update_traces(marker_line_width=0)
            st.plotly_chart(fig_fi, use_container_width=True)

    # PR Curve
    st.markdown('<div class="section-header">📉 Precision-Recall Tradeoff</div>', unsafe_allow_html=True)
    if metrics and 'attack_probabilities' in metrics:
        probs = metrics['attack_probabilities']
        y_true_arr = metrics['y_true']
        pr_data = []
        for t in range(5, 100, 2):
            preds = (probs * 100 >= t).astype(int)
            tp_t = ((y_true_arr == 1) & (preds == 1)).sum()
            fp_t = ((y_true_arr == 0) & (preds == 1)).sum()
            fn_t = ((y_true_arr == 1) & (preds == 0)).sum()
            p = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
            r = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0
            pr_data.append({'Threshold': t, 'Precision': p, 'Recall': r, 'F1': f})
        pr_df = pd.DataFrame(pr_data)

        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=pr_df['Threshold'], y=pr_df['Precision'],
                                    name='Precision', line=dict(color=ACCENT, width=2)))
        fig_pr.add_trace(go.Scatter(x=pr_df['Threshold'], y=pr_df['Recall'],
                                    name='Recall', line=dict(color=ACCENT_2, width=2)))
        fig_pr.add_trace(go.Scatter(x=pr_df['Threshold'], y=pr_df['F1'],
                                    name='F1', line=dict(color=SUCCESS, width=2, dash='dot')))
        apply_plotly_theme(fig_pr)
        fig_pr.update_layout(
            height=280,
            xaxis_title="Risk Score Threshold",
            yaxis_title="", yaxis_tickformat='.0%',
            legend=dict(orientation='h', y=1.1, x=0),
        )
        st.plotly_chart(fig_pr, use_container_width=True)


# ─────────────────────────────────────────────────────────
# TAB 5: RESEARCH QUESTION
# ─────────────────────────────────────────────────────────
with tab_research:
    st.markdown("""
    <div class="data-panel">
        <div class="data-panel-title">📋 Research Question</div>
        <div style="font-size:1rem; color:#e2e8f0; font-weight:500;">
            How effective is behavioral ML-based detection compared to rule-based baselines for identifying identity threats?
        </div>
    </div>
    """, unsafe_allow_html=True)

    if metrics and 'comparison' in metrics:
        comparison = metrics['comparison']

        # Comparison chart
        col_chart, col_table = st.columns([3, 2])

        with col_chart:
            st.markdown('<div class="section-header">Approach Comparison</div>', unsafe_allow_html=True)
            comp_df = pd.DataFrame(comparison)
            fig_comp = go.Figure()
            metric_names = ['precision', 'recall', 'f1_score']
            colors_comp = [ACCENT, ACCENT_2, SUCCESS]
            for i, approach in enumerate(comp_df['approach']):
                vals = [comp_df.iloc[i][m] for m in metric_names]
                fig_comp.add_trace(go.Bar(
                    name=approach,
                    x=['Precision', 'Recall', 'F1 Score'],
                    y=vals,
                    marker_color=colors_comp[i],
                    marker_line_width=0,
                ))
            apply_plotly_theme(fig_comp)
            fig_comp.update_layout(
                height=380, barmode='group',
                yaxis_tickformat='.0%', yaxis_title="",
                legend=dict(orientation='h', y=1.1, x=0),
            )
            st.plotly_chart(fig_comp, use_container_width=True)

        with col_table:
            st.markdown('<div class="section-header">Detailed Metrics</div>', unsafe_allow_html=True)
            display_comp = comp_df[['approach', 'accuracy', 'precision', 'recall', 'f1_score', 'fpr']].copy()
            display_comp.columns = ['Approach', 'Accuracy', 'Precision', 'Recall', 'F1', 'FPR']
            for col in ['Accuracy', 'Precision', 'Recall', 'F1', 'FPR']:
                display_comp[col] = (display_comp[col] * 100).round(1).astype(str) + '%'
            st.dataframe(display_comp, use_container_width=True, hide_index=True, height=180)

            # Key finding
            rules_m, ml_m, combined_m = comparison
            best = max(comparison, key=lambda x: x['f1_score'])
            st.markdown(f"""
            <div class="data-panel" style="margin-top: 14px;">
                <div class="data-panel-title">🏆 Key Finding</div>
                <div style="color:{TEXT_PRIMARY}; font-size: 0.88rem;">
                    <strong>{best['approach']}</strong> achieves the highest F1 score at
                    <span style="color:{ACCENT}; font-weight:700;">{best['f1_score']:.1%}</span>.
                </div>
                <div style="color:{TEXT_MUTED}; font-size: 0.8rem; margin-top: 6px;">
                    ML F1: {ml_m['f1_score']:.1%} · Rules F1: {rules_m['f1_score']:.1%} · Combined F1: {combined_m['f1_score']:.1%}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Roadmap
    st.markdown('<div class="section-header" style="margin-top: 24px;">Development Roadmap</div>', unsafe_allow_html=True)
    col_done, col_next = st.columns(2)
    with col_done:
        st.markdown(f"""
        <div class="data-panel">
            <div class="data-panel-title">✅ Completed (v2.0)</div>
            <div class="roadmap-item"><span class="roadmap-check">✓</span><span class="roadmap-text">Ensemble ML classifier (RF + HistGBT) — 95.4% accuracy</span></div>
            <div class="roadmap-item"><span class="roadmap-check">✓</span><span class="roadmap-text">Temporal train/test split — no data leakage</span></div>
            <div class="roadmap-item"><span class="roadmap-check">✓</span><span class="roadmap-text">24-feature behavioral extraction pipeline</span></div>
            <div class="roadmap-item"><span class="roadmap-check">✓</span><span class="roadmap-text">7 rule-based detections with MITRE ATT&CK mapping</span></div>
            <div class="roadmap-item"><span class="roadmap-check">✓</span><span class="roadmap-text">Automated response engine with playbook actions</span></div>
            <div class="roadmap-item"><span class="roadmap-check">✓</span><span class="roadmap-text">Per-alert ML explainability for SOC analysts</span></div>
            <div class="roadmap-item"><span class="roadmap-check">✓</span><span class="roadmap-text">Live streaming simulation pipeline</span></div>
            <div class="roadmap-item"><span class="roadmap-check">✓</span><span class="roadmap-text">Research comparison: Rules vs ML vs Combined</span></div>
        </div>
        """, unsafe_allow_html=True)
    with col_next:
        st.markdown(f"""
        <div class="data-panel">
            <div class="data-panel-title">🔜 Final Phase (v3.0)</div>
            <div class="roadmap-item"><span class="roadmap-pending">○</span><span class="roadmap-text">SIEM integration (Splunk/Sentinel export)</span></div>
            <div class="roadmap-item"><span class="roadmap-pending">○</span><span class="roadmap-text">Database-backed storage (replace CSV)</span></div>
            <div class="roadmap-item"><span class="roadmap-pending">○</span><span class="roadmap-text">Async real-time event processing</span></div>
            <div class="roadmap-item"><span class="roadmap-pending">○</span><span class="roadmap-text">UEBA user behavior profiling</span></div>
            <div class="roadmap-item"><span class="roadmap-pending">○</span><span class="roadmap-text">Model drift detection for production</span></div>
            <div class="roadmap-item"><span class="roadmap-pending">○</span><span class="roadmap-text">Expand MITRE ATT&CK technique coverage</span></div>
            <div class="roadmap-item"><span class="roadmap-pending">○</span><span class="roadmap-text">Final paper with full 30M-event benchmark</span></div>
            <div class="roadmap-item"><span class="roadmap-pending">○</span><span class="roadmap-text">Deployment guide and operational runbook</span></div>
        </div>
        """, unsafe_allow_html=True)
