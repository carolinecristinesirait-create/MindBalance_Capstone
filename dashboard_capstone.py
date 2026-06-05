
# =========================================================
# MindBalance Dashboard - Maroon & White Edition
# File utama untuk Streamlit Cloud: dashboard_capstone.py
# =========================================================

from pathlib import Path
import os

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import tensorflow as tf
except Exception as import_error:
    tf = None
    TF_IMPORT_ERROR = import_error
else:
    TF_IMPORT_ERROR = None


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="MindBalance | Anxiety Detection Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# GLOBAL STYLE
# =========================================================
MAROON = "#7A1F2B"
MAROON_DARK = "#4A0F18"
MAROON_SOFT = "#A64B59"
MAROON_LIGHT = "#F8E9EB"
CREAM = "#FFF8F5"
WHITE = "#FFFFFF"
INK = "#281B1D"
MUTED = "#7D6A6E"
ROSE = "#F2C9CF"
GOLD = "#C99A3D"
GREEN = "#2E7D5B"
ORANGE = "#B8672A"
RED = "#9C1D2E"


CUSTOM_CSS = f"""
<style>
    :root {{
        --maroon: {MAROON};
        --maroon-dark: {MAROON_DARK};
        --maroon-soft: {MAROON_SOFT};
        --maroon-light: {MAROON_LIGHT};
        --cream: {CREAM};
        --white: {WHITE};
        --ink: {INK};
        --muted: {MUTED};
        --gold: {GOLD};
    }}

    .stApp {{
        background: linear-gradient(135deg, #fff8f5 0%, #fff1f2 42%, #f8e9eb 100%);
        color: var(--ink);
    }}

    h1, h2, h3, h4, h5, h6 {{
        color: var(--maroon-dark) !important;
        letter-spacing: -0.02em;
    }}

    p, li, span, label, div {{
        font-family: "Inter", "Segoe UI", sans-serif;
    }}

    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #4a0f18 0%, #7a1f2b 58%, #9d3141 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.12);
    }}

    section[data-testid="stSidebar"] * {{
        color: #fff7f7 !important;
    }}

    section[data-testid="stSidebar"] [data-testid="stSelectbox"] div,
    section[data-testid="stSidebar"] [data-testid="stMultiSelect"] div,
    section[data-testid="stSidebar"] [data-baseweb="select"] * {{
        color: #2d1117 !important;
    }}

    .block-container {{
        padding-top: 2.0rem;
        padding-bottom: 4rem;
        max-width: 1280px;
    }}

    .hero-card {{
        background: radial-gradient(circle at top left, rgba(255,255,255,.38), rgba(255,255,255,0) 35%),
                    linear-gradient(135deg, #4a0f18 0%, #7a1f2b 50%, #a64b59 100%);
        color: #fff;
        padding: 2.3rem 2.4rem;
        border-radius: 28px;
        box-shadow: 0 22px 55px rgba(74, 15, 24, 0.28);
        border: 1px solid rgba(255,255,255,.16);
        margin-bottom: 1.4rem;
    }}

    .hero-title {{
        font-size: clamp(2rem, 4vw, 3.3rem);
        line-height: 1.05;
        font-weight: 850;
        color: #fff !important;
        margin-bottom: .7rem;
    }}

    .hero-subtitle {{
        font-size: 1.05rem;
        line-height: 1.75;
        color: rgba(255,255,255,.88);
        max-width: 850px;
        margin-bottom: 1.1rem;
    }}

    .hero-pill-row {{
        display: flex;
        flex-wrap: wrap;
        gap: .65rem;
        margin-top: 1.3rem;
    }}

    .hero-pill {{
        background: rgba(255,255,255,.15);
        color: #fff;
        border: 1px solid rgba(255,255,255,.22);
        border-radius: 999px;
        padding: .48rem .85rem;
        font-size: .88rem;
        font-weight: 650;
        backdrop-filter: blur(10px);
    }}

    .section-title {{
        display: flex;
        align-items: center;
        gap: .75rem;
        margin: 1.3rem 0 .8rem 0;
        padding-top: .45rem;
    }}

    .section-title .icon {{
        width: 42px;
        height: 42px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #7a1f2b, #b94d5f);
        color: #fff;
        border-radius: 14px;
        box-shadow: 0 10px 25px rgba(122,31,43,.22);
    }}

    .section-title .text {{
        font-size: 1.6rem;
        font-weight: 820;
        color: #4a0f18;
    }}

    .mini-note {{
        color: #7d6a6e;
        font-size: .95rem;
        margin-top: -.25rem;
        margin-bottom: 1rem;
    }}

    .card {{
        background: rgba(255,255,255,.92);
        border: 1px solid rgba(122,31,43,.12);
        border-radius: 22px;
        padding: 1.18rem 1.25rem;
        box-shadow: 0 14px 35px rgba(74, 15, 24, 0.08);
        margin-bottom: 1rem;
    }}

    .metric-card {{
        background: linear-gradient(180deg, #ffffff 0%, #fff7f7 100%);
        border: 1px solid rgba(122,31,43,.14);
        border-radius: 22px;
        padding: 1.18rem 1.25rem;
        min-height: 145px;
        box-shadow: 0 14px 35px rgba(74, 15, 24, 0.08);
        position: relative;
        overflow: hidden;
    }}

    .metric-card::after {{
        content: "";
        position: absolute;
        right: -45px;
        top: -45px;
        width: 120px;
        height: 120px;
        border-radius: 50%;
        background: rgba(122,31,43,.08);
    }}

    .metric-label {{
        color: #7d6a6e;
        font-size: .82rem;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: .05em;
        margin-bottom: .35rem;
    }}

    .metric-value {{
        color: #4a0f18;
        font-size: 2.05rem;
        line-height: 1.05;
        font-weight: 850;
        margin-bottom: .35rem;
    }}

    .metric-help {{
        color: #7d6a6e;
        font-size: .92rem;
        line-height: 1.45;
    }}

    .status-good {{ border-left: 7px solid #2e7d5b; }}
    .status-mid {{ border-left: 7px solid #c99a3d; }}
    .status-high {{ border-left: 7px solid #9c1d2e; }}

    .result-card {{
        padding: 1.6rem 1.7rem;
        border-radius: 24px;
        border: 1px solid rgba(122,31,43,.14);
        background: #fff;
        box-shadow: 0 18px 45px rgba(74, 15, 24, 0.10);
        margin: 1rem 0;
    }}

    .result-low {{
        background: linear-gradient(135deg, rgba(46,125,91,.12), #ffffff);
        border-left: 9px solid #2e7d5b;
    }}

    .result-medium {{
        background: linear-gradient(135deg, rgba(201,154,61,.15), #ffffff);
        border-left: 9px solid #c99a3d;
    }}

    .result-high {{
        background: linear-gradient(135deg, rgba(156,29,46,.15), #ffffff);
        border-left: 9px solid #9c1d2e;
    }}

    .result-title {{
        color: #4a0f18;
        font-size: 1.5rem;
        font-weight: 850;
        margin-bottom: .45rem;
    }}

    .result-desc {{
        color: #5c464b;
        font-size: 1rem;
        line-height: 1.65;
    }}

    .chip {{
        display: inline-block;
        padding: .35rem .7rem;
        border-radius: 999px;
        background: #f8e9eb;
        color: #7a1f2b;
        border: 1px solid rgba(122,31,43,.15);
        font-weight: 700;
        font-size: .82rem;
        margin: .15rem .22rem .15rem 0;
    }}

    .footer-box {{
        background: #4a0f18;
        color: rgba(255,255,255,.9);
        border-radius: 22px;
        padding: 1.2rem 1.35rem;
        margin-top: 2rem;
        font-size: .93rem;
        line-height: 1.6;
    }}

    div[data-testid="stMetric"] {{
        background: #fff;
        border: 1px solid rgba(122,31,43,.12);
        border-radius: 18px;
        padding: .95rem 1rem;
        box-shadow: 0 10px 25px rgba(74,15,24,.06);
    }}

    div[data-testid="stMetricValue"] {{
        color: #7a1f2b;
        font-weight: 850;
    }}

    .stButton button {{
        background: linear-gradient(135deg, #7a1f2b, #a64b59);
        color: white;
        border: 0;
        border-radius: 14px;
        padding: .75rem 1.1rem;
        font-weight: 800;
        box-shadow: 0 10px 26px rgba(122,31,43,.22);
    }}

    .stButton button:hover {{
        background: linear-gradient(135deg, #4a0f18, #7a1f2b);
        color: white;
        border: 0;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: .45rem;
        background: rgba(255,255,255,.65);
        padding: .55rem;
        border-radius: 18px;
        border: 1px solid rgba(122,31,43,.10);
    }}

    .stTabs [data-baseweb="tab"] {{
        border-radius: 14px;
        padding: .55rem .9rem;
        color: #7a1f2b;
        font-weight: 750;
    }}

    .stTabs [aria-selected="true"] {{
        background: #7a1f2b !important;
        color: #fff !important;
    }}

    hr {{
        border: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(122,31,43,.25), transparent);
        margin: 1.7rem 0;
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =========================================================
# PATH & FILE HELPERS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent


def find_first_existing_file(candidates, recursive_pattern=None):
    """Cari file dari beberapa kemungkinan lokasi agar aman saat deploy."""
    for path in candidates:
        path = Path(path)
        if path.exists() and path.is_file():
            return path

    if recursive_pattern:
        matches = list(BASE_DIR.rglob(recursive_pattern))
        if matches:
            return matches[0]

    return None


def relative_path_text(path):
    try:
        return str(path.relative_to(BASE_DIR))
    except Exception:
        return str(path)


# =========================================================
# SMALL UI HELPERS
# =========================================================
def section_header(icon, title, subtitle=None):
    st.markdown(
        f"""
        <div class="section-title">
            <div class="icon">{icon}</div>
            <div class="text">{title}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if subtitle:
        st.markdown(f"<div class='mini-note'>{subtitle}</div>", unsafe_allow_html=True)


def metric_card(label, value, help_text="", accent_class=""):
    st.markdown(
        f"""
        <div class="metric-card {accent_class}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_card(title, body, chips=None):
    chips_html = ""
    if chips:
        chips_html = "".join([f"<span class='chip'>{chip}</span>" for chip in chips])
    st.markdown(
        f"""
        <div class="card">
            <h4 style="margin-top:0; margin-bottom:.45rem; color:#4A0F18;">{title}</h4>
            <div style="color:#5c464b; line-height:1.65; font-size:.97rem;">{body}</div>
            <div style="margin-top:.8rem;">{chips_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def safe_percent(value):
    if pd.isna(value):
        return "0.0%"
    return f"{value:.1f}%"


def fig_to_streamlit(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# =========================================================
# DATA PREPARATION
# =========================================================
def normalize_name(name):
    """Ubah nama kolom menjadi format pembanding yang konsisten."""
    return (
        str(name)
        .strip()
        .lower()
        .replace("(", "")
        .replace(")", "")
        .replace("/", " ")
        .replace("-", " ")
        .replace("_", " ")
        .replace(".", " ")
    )


def ensure_column(df, canonical_name, aliases):
    """Buat kolom standar dari salah satu alias yang tersedia."""
    normalized_map = {normalize_name(col): col for col in df.columns}

    for alias in [canonical_name] + aliases:
        key = normalize_name(alias)
        if key in normalized_map:
            df[canonical_name] = df[normalized_map[key]]
            return df

    return df


def prepare_dataset(df):
    """Samakan nama kolom dataset agar cocok dengan visualisasi dan prediksi."""
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    alias_map = {
        "Age": ["age", "Umur"],
        "Gender": ["Jenis Kelamin", "Sex"],
        "Occupation": ["Pekerjaan", "Job", "Profession"],
        "Anxiety_Category": [
            "Anxiety Category",
            "Anxiety_Level_Category",
            "Anxiety Level Category",
            "Category",
            "Anxiety Category Label",
        ],
        "Sleep_Duration": [
            "Sleep Hours",
            "Sleep_Hours",
            "Sleep Duration",
            "Sleep_Duration_Hours",
            "Sleep Hours Daily",
        ],
        "Caffeine_Intake": [
            "Caffeine Intake (mg/day)",
            "Caffeine Intake",
            "Caffeine_Intake_mg_day",
            "Caffeine",
        ],
        "Stress_Level": [
            "Stress Level (1-10)",
            "Stress Level",
            "Stress_Level_1_10",
        ],
        "Heart_Rate": [
            "Heart Rate (bpm)",
            "Heart Rate",
            "Heart_Rate_bpm",
        ],
        "Breathing_Rate": [
            "Breathing Rate (breaths/min)",
            "Breathing Rate",
            "Breathing_Rate_breaths_min",
        ],
        "Sweating_Level": [
            "Sweating Level (1-5)",
            "Sweating Level",
            "Sweating_Level_1_5",
        ],
        "Physical_Activity": [
            "Physical Activity (hrs/week)",
            "Physical Activity",
            "Physical_Activity_hrs_week",
        ],
        "Alcohol_Consumption": [
            "Alcohol Consumption (drinks/week)",
            "Alcohol Consumption",
            "Alcohol",
        ],
        "Diet_Quality": [
            "Diet Quality (1-10)",
            "Diet Quality",
            "Diet_Quality_1_10",
        ],
        "Anxiety_Level": [
            "Anxiety Level (1-10)",
            "Anxiety Level",
            "Anxiety_Score",
        ],
        "Smoking": ["smoking", "Merokok"],
        "Family_History": [
            "Family History of Anxiety",
            "Family History",
            "Riwayat Keluarga",
        ],
        "Dizziness": ["dizziness", "Pusing"],
        "Medication": ["medication", "Obat", "Anxiety Medication"],
        "Therapy_Sessions": [
            "Therapy Sessions (per month)",
            "Therapy Sessions",
            "Sesi Terapi",
        ],
        "Recent_Life_Event": [
            "Recent Major Life Event",
            "Recent Life Event",
            "Major Life Event",
        ],
    }

    for canonical, aliases in alias_map.items():
        df = ensure_column(df, canonical, aliases)

    numeric_cols = [
        "Age",
        "Sleep_Duration",
        "Caffeine_Intake",
        "Stress_Level",
        "Heart_Rate",
        "Breathing_Rate",
        "Sweating_Level",
        "Physical_Activity",
        "Alcohol_Consumption",
        "Diet_Quality",
        "Anxiety_Level",
        "Therapy_Sessions",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    text_cols = [
        "Gender",
        "Occupation",
        "Anxiety_Category",
        "Smoking",
        "Family_History",
        "Dizziness",
        "Medication",
        "Recent_Life_Event",
    ]

    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    if "Anxiety_Category" in df.columns:
        category_fix = {
            "low": "Low",
            "medium": "Medium",
            "moderate": "Medium",
            "high": "High",
        }
        df["Anxiety_Category"] = (
            df["Anxiety_Category"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(category_fix)
            .fillna(df["Anxiety_Category"].astype(str).str.strip())
        )

    return df


@st.cache_data(show_spinner=False)
def load_data():
    data_path = find_first_existing_file(
        candidates=[
            BASE_DIR / "data" / "cleaned_anxiety_data.csv",
            BASE_DIR / "cleaned_anxiety_data.csv",
        ],
        recursive_pattern="cleaned_anxiety_data*.csv",
    )

    if data_path is None:
        raise FileNotFoundError(
            "File dataset tidak ditemukan. Simpan dataset di 'data/cleaned_anxiety_data.csv'."
        )

    df = pd.read_csv(data_path)
    df = prepare_dataset(df)
    return df, data_path


# =========================================================
# MODEL LOADING & PREDICTION
# =========================================================
@st.cache_resource(show_spinner=False)
def load_ai_model():
    if tf is None:
        raise RuntimeError(f"TensorFlow gagal di-import: {TF_IMPORT_ERROR}")

    model_path = find_first_existing_file(
        candidates=[
            BASE_DIR / "models" / "mindbalance_model_new.keras",
            BASE_DIR / "mindbalance_model_new.keras",
            BASE_DIR / "models" / "mindbalance_model_new(1).keras",
            BASE_DIR / "mindbalance_model_new(1).keras",
        ],
        recursive_pattern="mindbalance_model_new*.keras",
    )

    if model_path is None:
        raise FileNotFoundError(
            "File model tidak ditemukan. Simpan model di 'models/mindbalance_model_new.keras'."
        )

    return tf.keras.models.load_model(str(model_path), compile=False), model_path


def fallback_prediction(anxiety_composite, stress_level, sleep_hours, caffeine):
    """Mode cadangan kalau model AI gagal dimuat."""
    extra_risk = 0
    if sleep_hours < 6:
        extra_risk += 0.08
    if caffeine > 350:
        extra_risk += 0.05

    score = anxiety_composite + extra_risk

    if score >= 0.55 or stress_level >= 8:
        return 2, "High (Tinggi)"
    if score >= 0.33 or stress_level >= 5:
        return 1, "Medium (Sedang)"
    return 0, "Low (Rendah)"


def predict_with_model(model, input_vector):
    outputs = model(tf.constant(input_vector), training=False)

    if isinstance(outputs, (list, tuple)):
        class_out = outputs[0]
    elif isinstance(outputs, dict):
        class_out = outputs.get("class_output", list(outputs.values())[0])
    else:
        class_out = outputs

    class_values = class_out.numpy()[0] if hasattr(class_out, "numpy") else np.array(class_out)[0]
    predicted_class_idx = int(np.argmax(class_values))
    confidence = float(np.max(class_values)) if len(class_values) > 1 else np.nan

    labels = ["Low (Rendah)", "Medium (Sedang)", "High (Tinggi)"]
    return predicted_class_idx, labels[predicted_class_idx], confidence, class_values


# =========================================================
# LOAD RESOURCES
# =========================================================
try:
    df, data_path_used = load_data()
    data_loaded = True
except Exception as data_error:
    data_loaded = False
    data_path_used = None
    df = pd.DataFrame(
        {
            "Anxiety_Category": ["Low", "Medium", "High", "Low", "High", "Medium"],
            "Sleep_Duration": [7.5, 6.0, 5.0, 8.0, 4.5, 6.5],
            "Caffeine_Intake": [100, 250, 400, 50, 450, 200],
            "Stress_Level": [3, 6, 9, 2, 8, 5],
            "Heart_Rate": [70, 82, 95, 68, 90, 78],
            "Physical_Activity": [3, 4, 1, 5, 1, 3],
            "Diet_Quality": [7, 6, 4, 8, 3, 6],
            "Age": [23, 26, 30, 21, 34, 28],
        }
    )

try:
    model, model_path_used = load_ai_model()
    model_loaded = True
except Exception as model_error:
    model = None
    model_path_used = None
    model_loaded = False


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown(
        """
        <div style="padding:.8rem 0 1.2rem 0;">
            <div style="font-size:1.65rem; font-weight:900; line-height:1.05; color:#fff;">🧠 MindBalance</div>
            <div style="font-size:.9rem; color:rgba(255,255,255,.78); margin-top:.35rem; line-height:1.45;">
                AI-powered anxiety detection & mental wellness dashboard.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigasi Dashboard",
        [
            "🏠 Executive Overview",
            "📊 Dataset Overview",
            "🌙 Lifestyle Insights",
            "🫀 Physical Indicators",
            "🔗 Correlation Explorer",
            "🔮 AI Detection Form",
            "ℹ️ About & Methodology",
        ],
    )

    st.markdown("---")
    st.markdown("### Status File")
    if data_loaded:
        st.success(f"Dataset: {relative_path_text(data_path_used)}")
    else:
        st.warning("Dataset memakai dummy data.")

    if model_loaded:
        st.success(f"Model: {relative_path_text(model_path_used)}")
    else:
        st.warning("Model AI belum aktif. Fallback aktif.")

    st.markdown("---")
    st.caption("Gunakan dashboard ini sebagai deteksi awal/simulasi, bukan diagnosis medis final.")


# =========================================================
# GLOBAL FILTERS FOR EDA PAGES
# =========================================================
def get_filtered_data(df):
    filtered = df.copy()

    with st.expander("🔎 Filter Data", expanded=False):
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            if "Anxiety_Category" in filtered.columns:
                order = ["Low", "Medium", "High"]
                options = [x for x in order if x in filtered["Anxiety_Category"].dropna().unique().tolist()]
                options += [x for x in sorted(filtered["Anxiety_Category"].dropna().unique().tolist()) if x not in options]
                chosen = st.multiselect("Kategori kecemasan", options=options, default=options)
                if chosen:
                    filtered = filtered[filtered["Anxiety_Category"].isin(chosen)]

        with col_b:
            if "Gender" in filtered.columns:
                gender_options = sorted(filtered["Gender"].dropna().unique().tolist())
                chosen_gender = st.multiselect("Gender", options=gender_options, default=gender_options)
                if chosen_gender:
                    filtered = filtered[filtered["Gender"].isin(chosen_gender)]

        with col_c:
            if "Occupation" in filtered.columns:
                occupation_options = sorted(filtered["Occupation"].dropna().unique().tolist())
                chosen_occ = st.multiselect("Pekerjaan", options=occupation_options, default=occupation_options)
                if chosen_occ:
                    filtered = filtered[filtered["Occupation"].isin(chosen_occ)]

        if "Age" in filtered.columns and filtered["Age"].notna().any():
            min_age = int(filtered["Age"].min())
            max_age = int(filtered["Age"].max())
            if min_age < max_age:
                age_range = st.slider("Rentang umur", min_age, max_age, (min_age, max_age))
                filtered = filtered[(filtered["Age"] >= age_range[0]) & (filtered["Age"] <= age_range[1])]

    if filtered.empty:
        st.warning("Tidak ada data sesuai filter. Silakan ubah filter.")
        st.stop()

    return filtered


# =========================================================
# PLOTTING FUNCTIONS
# =========================================================
def plot_count_by_category(data):
    order = ["Low", "Medium", "High"]
    available_order = [cat for cat in order if cat in data["Anxiety_Category"].dropna().unique()]
    palette = {"Low": GREEN, "Medium": GOLD, "High": RED}

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    sns.countplot(
        data=data,
        x="Anxiety_Category",
        order=available_order,
        palette=palette,
        ax=ax,
    )
    ax.set_title("Distribusi Kategori Kecemasan", fontsize=14, fontweight="bold", color=MAROON_DARK)
    ax.set_xlabel("Kategori Kecemasan")
    ax.set_ylabel("Jumlah Pengguna")
    ax.grid(axis="y", alpha=0.18)
    sns.despine(ax=ax)
    return fig


def plot_box(data, y_col, title, ylabel):
    order = ["Low", "Medium", "High"]
    available_order = [cat for cat in order if cat in data["Anxiety_Category"].dropna().unique()]
    palette = {"Low": "#DCEFE7", "Medium": "#F7E7BD", "High": "#F2C9CF"}

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    sns.boxplot(
        data=data,
        x="Anxiety_Category",
        y=y_col,
        order=available_order,
        palette=palette,
        ax=ax,
    )
    sns.stripplot(
        data=data.sample(min(len(data), 350), random_state=42),
        x="Anxiety_Category",
        y=y_col,
        order=available_order,
        color=MAROON_DARK,
        alpha=0.22,
        size=2.5,
        ax=ax,
    )
    ax.set_title(title, fontsize=14, fontweight="bold", color=MAROON_DARK)
    ax.set_xlabel("Kategori Kecemasan")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.18)
    sns.despine(ax=ax)
    return fig


def plot_bar_mean(data, x_col, y_col, title, xlabel, ylabel, top_n=None):
    temp = data[[x_col, y_col]].dropna().copy()
    if top_n:
        counts = temp[x_col].value_counts().head(top_n).index
        temp = temp[temp[x_col].isin(counts)]

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    sns.barplot(data=temp, x=x_col, y=y_col, estimator="mean", errorbar=None, color=MAROON_SOFT, ax=ax)
    ax.set_title(title, fontsize=14, fontweight="bold", color=MAROON_DARK)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.18)
    ax.tick_params(axis="x", rotation=25)
    sns.despine(ax=ax)
    return fig


def plot_scatter(data, x_col, y_col, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    hue = "Anxiety_Category" if "Anxiety_Category" in data.columns else None
    palette = {"Low": GREEN, "Medium": GOLD, "High": RED}
    sns.scatterplot(
        data=data,
        x=x_col,
        y=y_col,
        hue=hue,
        palette=palette if hue else None,
        alpha=0.55,
        s=34,
        ax=ax,
    )
    ax.set_title(title, fontsize=14, fontweight="bold", color=MAROON_DARK)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.16)
    sns.despine(ax=ax)
    return fig


def plot_heatmap(data):
    numeric_df = data.select_dtypes(include=np.number).copy()
    numeric_df = numeric_df.dropna(axis=1, how="all")

    if numeric_df.shape[1] < 2:
        return None

    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.heatmap(
        corr,
        annot=True,
        cmap="rocket_r",
        fmt=".2f",
        linewidths=0.5,
        linecolor="#fff4f4",
        ax=ax,
        cbar_kws={"shrink": 0.85},
    )
    ax.set_title("Matriks Korelasi Fitur Numerik", fontsize=15, fontweight="bold", color=MAROON_DARK)
    return fig


def plot_distribution(data, col, title, xlabel):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    sns.histplot(data=data, x=col, kde=True, color=MAROON, ax=ax)
    ax.set_title(title, fontsize=14, fontweight="bold", color=MAROON_DARK)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frekuensi")
    ax.grid(axis="y", alpha=0.18)
    sns.despine(ax=ax)
    return fig


# =========================================================
# SHARED KPI CALCULATION
# =========================================================
def compute_kpis(data):
    total = len(data)
    high_pct = 0
    medium_pct = 0
    low_pct = 0

    if "Anxiety_Category" in data.columns and total > 0:
        counts = data["Anxiety_Category"].value_counts(normalize=True) * 100
        high_pct = counts.get("High", 0)
        medium_pct = counts.get("Medium", 0)
        low_pct = counts.get("Low", 0)

    avg_sleep = data["Sleep_Duration"].mean() if "Sleep_Duration" in data.columns else np.nan
    avg_stress = data["Stress_Level"].mean() if "Stress_Level" in data.columns else np.nan
    avg_caffeine = data["Caffeine_Intake"].mean() if "Caffeine_Intake" in data.columns else np.nan
    avg_anxiety = data["Anxiety_Level"].mean() if "Anxiety_Level" in data.columns else np.nan

    return {
        "total": total,
        "high_pct": high_pct,
        "medium_pct": medium_pct,
        "low_pct": low_pct,
        "avg_sleep": avg_sleep,
        "avg_stress": avg_stress,
        "avg_caffeine": avg_caffeine,
        "avg_anxiety": avg_anxiety,
    }


def render_kpi_row(data):
    kpi = compute_kpis(data)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total Records", f"{kpi['total']:,}", "Jumlah baris data yang dianalisis.")
    with c2:
        metric_card("High Anxiety", safe_percent(kpi["high_pct"]), "Proporsi data dengan kategori kecemasan tinggi.", "status-high")
    with c3:
        metric_card("Avg Stress", f"{kpi['avg_stress']:.2f}" if not pd.isna(kpi["avg_stress"]) else "-", "Rata-rata stress level skala 1–10.", "status-mid")
    with c4:
        metric_card("Avg Sleep", f"{kpi['avg_sleep']:.2f} jam" if not pd.isna(kpi["avg_sleep"]) else "-", "Rata-rata durasi tidur pengguna.", "status-good")


# =========================================================
# PAGE: EXECUTIVE OVERVIEW
# =========================================================
def page_executive_overview():
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">MindBalance Dashboard</div>
            <div class="hero-subtitle">
                Dashboard interaktif berwarna maroon-putih untuk mengeksplorasi pola gaya hidup,
                indikator fisik, dan prediksi awal tingkat kecemasan menggunakan model AI.
            </div>
            <div class="hero-pill-row">
                <span class="hero-pill">📊 Interactive EDA</span>
                <span class="hero-pill">🧠 AI Inference</span>
                <span class="hero-pill">🌙 Lifestyle Pattern</span>
                <span class="hero-pill">🫀 Physical Indicators</span>
                <span class="hero-pill">🔗 Correlation Explorer</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_header("📌", "Executive Summary", "Ringkasan kondisi dataset dan indikator utama yang paling cepat dibaca.")
    render_kpi_row(df)

    st.markdown("---")
    left, right = st.columns([1.1, 0.9])

    with left:
        if "Anxiety_Category" in df.columns:
            fig = plot_count_by_category(df)
            fig_to_streamlit(fig)
        else:
            st.warning("Kolom kategori kecemasan belum tersedia.")

    with right:
        info_card(
            "Apa yang bisa dibaca dari dashboard ini?",
            "Dashboard ini membagi analisis menjadi beberapa bagian: distribusi kategori kecemasan, hubungan tidur dan kafein, indikator fisik seperti detak jantung dan frekuensi napas, serta form prediksi AI untuk simulasi deteksi awal.",
            chips=["EDA", "Lifestyle", "Physical", "Prediction"],
        )
        info_card(
            "Status model",
            "Jika file model .keras berhasil dimuat, prediksi akan menggunakan model AI. Jika tidak, aplikasi otomatis memakai mode cadangan berbasis skor komposit agar tetap bisa berjalan.",
            chips=["TensorFlow", "Keras", "Fallback Safe"],
        )

    st.markdown("---")
    section_header("🧭", "Quick Navigation", "Pilih halaman di sidebar untuk melihat analisis yang lebih rinci.")
    c1, c2, c3 = st.columns(3)
    with c1:
        info_card("Lifestyle Insights", "Analisis durasi tidur, konsumsi kafein, aktivitas fisik, pola makan, dan hubungannya dengan kategori kecemasan.", ["Sleep", "Caffeine", "Diet"])
    with c2:
        info_card("Physical Indicators", "Analisis indikator fisik seperti heart rate, breathing rate, sweating level, dizziness, dan medication.", ["Heart Rate", "Breathing", "Sweating"])
    with c3:
        info_card("AI Detection Form", "Form input untuk menjalankan prediksi awal tingkat kecemasan berdasarkan 21 fitur input model.", ["Inference", "AI Model", "Score"])


# =========================================================
# PAGE: DATASET OVERVIEW
# =========================================================
def page_dataset_overview():
    section_header("📊", "Dataset Overview", "Gambaran umum struktur data, statistik ringkas, dan proporsi kategori.")
    filtered = get_filtered_data(df)
    render_kpi_row(filtered)

    tab1, tab2, tab3 = st.tabs(["📌 Struktur Data", "📈 Distribusi Utama", "🧾 Data Preview"])

    with tab1:
        c1, c2 = st.columns([0.9, 1.1])
        with c1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Informasi Dataset")
            st.write(f"Jumlah baris: **{len(filtered):,}**")
            st.write(f"Jumlah kolom: **{filtered.shape[1]:,}**")
            st.write(f"Jumlah fitur numerik: **{filtered.select_dtypes(include=np.number).shape[1]:,}**")
            st.write(f"Jumlah fitur kategorik: **{filtered.select_dtypes(exclude=np.number).shape[1]:,}**")
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.dataframe(
                pd.DataFrame(
                    {
                        "Kolom": filtered.columns,
                        "Tipe Data": [str(dtype) for dtype in filtered.dtypes],
                        "Missing Value": filtered.isna().sum().values,
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            if "Anxiety_Category" in filtered.columns:
                fig = plot_count_by_category(filtered)
                fig_to_streamlit(fig)
        with c2:
            if "Anxiety_Category" in filtered.columns:
                category_count = filtered["Anxiety_Category"].value_counts().reset_index()
                category_count.columns = ["Kategori", "Jumlah"]
                st.dataframe(category_count, use_container_width=True, hide_index=True)

        c3, c4 = st.columns(2)
        with c3:
            if "Age" in filtered.columns:
                fig = plot_distribution(filtered, "Age", "Distribusi Umur", "Umur")
                fig_to_streamlit(fig)
        with c4:
            if "Anxiety_Level" in filtered.columns:
                fig = plot_distribution(filtered, "Anxiety_Level", "Distribusi Anxiety Level", "Anxiety Level (1–10)")
                fig_to_streamlit(fig)

    with tab3:
        st.dataframe(filtered.head(100), use_container_width=True)
        with st.expander("Statistik deskriptif fitur numerik"):
            st.dataframe(filtered.describe().T, use_container_width=True)


# =========================================================
# PAGE: LIFESTYLE INSIGHTS
# =========================================================
def page_lifestyle_insights():
    section_header("🌙", "Lifestyle Insights", "Fokus pada tidur, kafein, aktivitas fisik, alkohol, dan kualitas pola makan.")
    filtered = get_filtered_data(df)
    render_kpi_row(filtered)

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["🌙 Sleep & Anxiety", "☕ Caffeine & Lifestyle", "🥗 Diet & Activity"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            if {"Anxiety_Category", "Sleep_Duration"}.issubset(filtered.columns):
                fig = plot_box(filtered, "Sleep_Duration", "Durasi Tidur Berdasarkan Kategori Kecemasan", "Durasi Tidur (Jam)")
                fig_to_streamlit(fig)
        with c2:
            if {"Sleep_Duration", "Stress_Level"}.issubset(filtered.columns):
                fig = plot_scatter(filtered, "Sleep_Duration", "Stress_Level", "Tidur vs Stress Level", "Durasi Tidur (Jam)", "Stress Level")
                fig_to_streamlit(fig)

        info_card(
            "Insight tidur",
            "Durasi tidur dapat menjadi salah satu indikator gaya hidup yang relevan untuk dibaca bersama stress level. Pola tidur yang lebih pendek sering menjadi sinyal awal untuk mengevaluasi kebiasaan harian dan manajemen stres.",
            ["Sleep Hygiene", "Stress", "Wellness"],
        )

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            if {"Anxiety_Category", "Caffeine_Intake"}.issubset(filtered.columns):
                fig = plot_box(filtered, "Caffeine_Intake", "Konsumsi Kafein Berdasarkan Kecemasan", "Kafein (mg/hari)")
                fig_to_streamlit(fig)
        with c2:
            if {"Caffeine_Intake", "Heart_Rate"}.issubset(filtered.columns):
                fig = plot_scatter(filtered, "Caffeine_Intake", "Heart_Rate", "Kafein vs Heart Rate", "Kafein (mg/hari)", "Heart Rate (BPM)")
                fig_to_streamlit(fig)

        if "Caffeine_Intake" in filtered.columns:
            caffeine_bins = pd.cut(
                filtered["Caffeine_Intake"],
                bins=[-1, 100, 250, 400, 700],
                labels=["Low", "Moderate", "High", "Very High"],
            )
            caffeine_summary = (
                filtered.assign(Caffeine_Level=caffeine_bins)
                .groupby("Caffeine_Level", observed=False)["Stress_Level"]
                .mean()
                .reset_index()
            ) if "Stress_Level" in filtered.columns else None

            if caffeine_summary is not None:
                st.dataframe(caffeine_summary, use_container_width=True, hide_index=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            if {"Anxiety_Category", "Physical_Activity"}.issubset(filtered.columns):
                fig = plot_box(filtered, "Physical_Activity", "Aktivitas Fisik Berdasarkan Kecemasan", "Jam/Minggu")
                fig_to_streamlit(fig)
        with c2:
            if {"Anxiety_Category", "Diet_Quality"}.issubset(filtered.columns):
                fig = plot_box(filtered, "Diet_Quality", "Kualitas Pola Makan Berdasarkan Kecemasan", "Skala 1–10")
                fig_to_streamlit(fig)

        if {"Occupation", "Stress_Level"}.issubset(filtered.columns):
            fig = plot_bar_mean(filtered, "Occupation", "Stress_Level", "Rata-rata Stress Level per Pekerjaan", "Pekerjaan", "Avg Stress Level", top_n=12)
            fig_to_streamlit(fig)


# =========================================================
# PAGE: PHYSICAL INDICATORS
# =========================================================
def page_physical_indicators():
    section_header("🫀", "Physical Indicators", "Analisis indikator fisik dan medis yang berkaitan dengan kondisi kecemasan.")
    filtered = get_filtered_data(df)
    render_kpi_row(filtered)

    tab1, tab2, tab3 = st.tabs(["❤️ Heart & Breathing", "💧 Sweating & Dizziness", "💊 Medication & Therapy"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            if {"Anxiety_Category", "Heart_Rate"}.issubset(filtered.columns):
                fig = plot_box(filtered, "Heart_Rate", "Heart Rate Berdasarkan Kategori Kecemasan", "BPM")
                fig_to_streamlit(fig)
        with c2:
            if {"Anxiety_Category", "Breathing_Rate"}.issubset(filtered.columns):
                fig = plot_box(filtered, "Breathing_Rate", "Breathing Rate Berdasarkan Kategori Kecemasan", "Napas/Menit")
                fig_to_streamlit(fig)

        if {"Heart_Rate", "Breathing_Rate"}.issubset(filtered.columns):
            fig = plot_scatter(filtered, "Heart_Rate", "Breathing_Rate", "Heart Rate vs Breathing Rate", "Heart Rate (BPM)", "Breathing Rate")
            fig_to_streamlit(fig)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            if {"Anxiety_Category", "Sweating_Level"}.issubset(filtered.columns):
                fig = plot_box(filtered, "Sweating_Level", "Sweating Level Berdasarkan Kecemasan", "Skala 1–5")
                fig_to_streamlit(fig)
        with c2:
            if {"Dizziness", "Anxiety_Category"}.issubset(filtered.columns):
                temp = filtered.groupby(["Dizziness", "Anxiety_Category"]).size().reset_index(name="Jumlah")
                fig, ax = plt.subplots(figsize=(7.2, 4.8))
                sns.barplot(data=temp, x="Dizziness", y="Jumlah", hue="Anxiety_Category", palette={"Low": GREEN, "Medium": GOLD, "High": RED}, ax=ax)
                ax.set_title("Dizziness dan Kategori Kecemasan", fontsize=14, fontweight="bold", color=MAROON_DARK)
                ax.set_xlabel("Dizziness")
                ax.set_ylabel("Jumlah")
                ax.grid(axis="y", alpha=.18)
                sns.despine(ax=ax)
                fig_to_streamlit(fig)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            if {"Medication", "Anxiety_Category"}.issubset(filtered.columns):
                temp = filtered.groupby(["Medication", "Anxiety_Category"]).size().reset_index(name="Jumlah")
                fig, ax = plt.subplots(figsize=(7.2, 4.8))
                sns.barplot(data=temp, x="Medication", y="Jumlah", hue="Anxiety_Category", palette={"Low": GREEN, "Medium": GOLD, "High": RED}, ax=ax)
                ax.set_title("Medication dan Kategori Kecemasan", fontsize=14, fontweight="bold", color=MAROON_DARK)
                ax.set_xlabel("Medication")
                ax.set_ylabel("Jumlah")
                ax.grid(axis="y", alpha=.18)
                sns.despine(ax=ax)
                fig_to_streamlit(fig)
        with c2:
            if {"Therapy_Sessions", "Anxiety_Category"}.issubset(filtered.columns):
                fig = plot_box(filtered, "Therapy_Sessions", "Sesi Terapi per Bulan Berdasarkan Kecemasan", "Sesi/Bulan")
                fig_to_streamlit(fig)

        info_card(
            "Catatan pembacaan indikator fisik",
            "Indikator fisik seperti heart rate, breathing rate, sweating, dan dizziness tidak dapat berdiri sendiri sebagai diagnosis. Indikator tersebut lebih tepat dibaca sebagai sinyal pendukung bersama konteks gaya hidup dan kondisi psikologis.",
            ["Not Diagnosis", "Early Screening", "Context Matters"],
        )


# =========================================================
# PAGE: CORRELATION EXPLORER
# =========================================================
def page_correlation_explorer():
    section_header("🔗", "Correlation Explorer", "Melihat hubungan antarfitur numerik dan memilih variabel untuk dibandingkan.")
    filtered = get_filtered_data(df)

    tab1, tab2, tab3 = st.tabs(["🔥 Heatmap", "🎯 Compare Variables", "📋 Correlation Table"])

    with tab1:
        fig = plot_heatmap(filtered)
        if fig is None:
            st.warning("Kolom numerik belum cukup untuk membuat heatmap.")
        else:
            fig_to_streamlit(fig)

    with tab2:
        numeric_cols = filtered.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) >= 2:
            c1, c2 = st.columns(2)
            with c1:
                x_col = st.selectbox("Pilih variabel X", numeric_cols, index=0)
            with c2:
                default_y = numeric_cols.index("Stress_Level") if "Stress_Level" in numeric_cols else min(1, len(numeric_cols) - 1)
                y_col = st.selectbox("Pilih variabel Y", numeric_cols, index=default_y)

            fig = plot_scatter(filtered, x_col, y_col, f"{x_col} vs {y_col}", x_col, y_col)
            fig_to_streamlit(fig)
        else:
            st.warning("Butuh minimal dua kolom numerik untuk comparison.")

    with tab3:
        numeric_df = filtered.select_dtypes(include=np.number)
        if numeric_df.shape[1] >= 2:
            corr_table = numeric_df.corr().round(3)
            st.dataframe(corr_table, use_container_width=True)
        else:
            st.warning("Kolom numerik belum cukup untuk tabel korelasi.")


# =========================================================
# PAGE: AI DETECTION FORM
# =========================================================
def page_ai_detection_form():
    section_header("🔮", "AI Anxiety Detection Form", "Masukkan profil dan kebiasaan harian untuk mendapatkan prediksi awal tingkat kecemasan.")

    c1, c2 = st.columns([1.1, 0.9])
    with c1:
        info_card(
            "Cara membaca hasil",
            "Hasil prediksi dikategorikan menjadi Low, Medium, atau High. Dashboard juga menampilkan tiga skor turunan: Sleep Efficiency Score, Lifestyle Risk Index, dan Anxiety Composite Score.",
            ["Low", "Medium", "High"],
        )
    with c2:
        if model_loaded:
            info_card("Model aktif", f"Model AI berhasil dimuat dari <b>{relative_path_text(model_path_used)}</b>.", ["Keras", "TensorFlow", "Active"])
        else:
            info_card("Fallback aktif", "Model AI belum berhasil dimuat, sehingga prediksi memakai mode cadangan berbasis skor komposit.", ["Fallback", "Rule-based", "Safe"])

    with st.form("prediction_form"):
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("👤 Profil Demografi & Kebiasaan")
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Umur", min_value=18, max_value=100, value=25)
            gender = st.selectbox("Jenis Kelamin", ["Male", "Female", "Other"])
            occupation = st.selectbox(
                "Pekerjaan",
                [
                    "Artist", "Athlete", "Chef", "Doctor", "Engineer",
                    "Freelancer", "Lawyer", "Musician", "Nurse", "Other",
                    "Scientist", "Student", "Teacher",
                ],
            )

        with col2:
            sleep_hours = st.slider("Durasi Tidur Harian (Jam)", 2.3, 11.3, 7.0, 0.1)
            physical_activity = st.slider("Aktivitas Fisik (Jam/Minggu)", 0.0, 10.1, 3.0, 0.1)
            caffeine = st.number_input("Konsumsi Kafein Harian (mg)", min_value=0, max_value=599, value=150)

        with col3:
            alcohol = st.number_input("Konsumsi Alkohol (Gelas/Minggu)", min_value=0, max_value=19, value=2)
            smoking = st.selectbox("Apakah Merokok?", ["Yes", "No"])
            diet_quality = st.slider("Kualitas Pola Makan (1–10)", 1, 10, 7)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🫀 Parameter Fisik & Medis")
        col4, col5, col6 = st.columns(3)

        with col4:
            stress_level = st.slider("Tingkat Stres Psikologis (1–10)", 1, 10, 5)
            heart_rate = st.number_input("Detak Jantung Istirahat (BPM)", min_value=60, max_value=119, value=75)
            breathing_rate = st.number_input("Frekuensi Napas (Napas/Menit)", min_value=12, max_value=29, value=18)

        with col5:
            sweating_level = st.slider("Tingkat Keringat Berlebih (1–5)", 1, 5, 2)
            dizziness = st.selectbox("Sering Merasa Pusing?", ["Yes", "No"])
            medication = st.selectbox("Sedang Konsumsi Obat Kecemasan?", ["Yes", "No"])

        with col6:
            therapy_sessions = st.number_input("Sesi Terapi/Konseling Bulan Ini", min_value=0, max_value=12, value=0)
            family_history = st.selectbox("Riwayat Kecemasan di Keluarga?", ["Yes", "No"])
            recent_life_event = st.selectbox("Kejadian Besar/Trauma 6 Bulan Terakhir?", ["Yes", "No"])
        st.markdown("</div>", unsafe_allow_html=True)

        submit_button = st.form_submit_button(label="🔍 Mulai Deteksi AI")

    if submit_button:
        st.markdown("---")
        section_header("📋", "Hasil Analisis", "Ringkasan prediksi dan rekomendasi coping awal.")

        sleep_norm = (sleep_hours - 2.3) / (11.3 - 2.3)
        caffeine_norm = caffeine / 599.0
        sleep_efficiency = round((sleep_norm * 0.7 + (1 - caffeine_norm) * 0.3), 4)

        binary_map = {"Yes": 1, "No": 0}
        stress_norm = (stress_level - 1) / 9
        alcohol_norm = alcohol / 19.0
        activity_norm = physical_activity / 10.1
        diet_norm = (diet_quality - 1) / 9

        lifestyle_risk = round(
            (
                stress_norm * 0.25
                + binary_map[smoking] * 0.10
                + alcohol_norm * 0.10
                + binary_map[family_history] * 0.15
                + binary_map[recent_life_event] * 0.15
                + (1 - activity_norm) * 0.10
                + (1 - diet_norm) * 0.15
            ),
            4,
        )

        hr_norm = (heart_rate - 60) / (119 - 60)
        br_norm = (breathing_rate - 12) / (29 - 12)
        sweat_norm = (sweating_level - 1) / 4

        anxiety_composite = round(
            (
                stress_norm * 0.30
                + hr_norm * 0.20
                + br_norm * 0.20
                + sweat_norm * 0.15
                + binary_map[family_history] * 0.15
            ),
            4,
        )

        gender_map = {"Male": 0, "Female": 1, "Other": 2}
        occ_list = [
            "Artist", "Athlete", "Chef", "Doctor", "Engineer",
            "Freelancer", "Lawyer", "Musician", "Nurse", "Other",
            "Scientist", "Student", "Teacher",
        ]
        occ_map = {occ: i for i, occ in enumerate(occ_list)}

        input_vector = np.array(
            [[
                age,
                gender_map[gender],
                occ_map[occupation],
                sleep_hours,
                physical_activity,
                caffeine,
                alcohol,
                binary_map[smoking],
                binary_map[family_history],
                stress_level,
                heart_rate,
                breathing_rate,
                sweating_level,
                binary_map[dizziness],
                binary_map[medication],
                therapy_sessions,
                binary_map[recent_life_event],
                diet_quality,
                sleep_efficiency,
                lifestyle_risk,
                anxiety_composite,
            ]],
            dtype=np.float32,
        )

        confidence = np.nan
        class_values = None
        used_fallback = False

        if model_loaded:
            try:
                with st.spinner("Model AI sedang memproses matriks klinis Anda..."):
                    predicted_class_idx, anxiety_result, confidence, class_values = predict_with_model(model, input_vector)
            except Exception as predict_error:
                used_fallback = True
                st.warning(f"Model AI gagal melakukan prediksi, memakai mode cadangan. Detail: {predict_error}")
                predicted_class_idx, anxiety_result = fallback_prediction(anxiety_composite, stress_level, sleep_hours, caffeine)
        else:
            used_fallback = True
            predicted_class_idx, anxiety_result = fallback_prediction(anxiety_composite, stress_level, sleep_hours, caffeine)

        result_class = ["result-low", "result-medium", "result-high"][predicted_class_idx]
        result_icon = ["🟢", "🟡", "🔴"][predicted_class_idx]

        if predicted_class_idx == 2:
            rec = "Ambil waktu jeda istirahat, batasi asupan kopi/kafein harian, lakukan teknik box breathing, dan sangat disarankan berdiskusi dengan psikolog atau tenaga profesional."
        elif predicted_class_idx == 1:
            rec = "Tingkatkan durasi tidur harian, luangkan waktu 15 menit untuk jalan santai/olahraga ringan, kurangi stimulan di malam hari, dan pantau pemicu stres."
        else:
            rec = "Tingkat kecemasan berada pada kategori rendah. Pertahankan pola tidur, aktivitas fisik, dan manajemen stres yang sudah baik."

        confidence_text = "Mode cadangan" if used_fallback or pd.isna(confidence) else f"Confidence model: {confidence:.2%}"

        st.markdown(
            f"""
            <div class="result-card {result_class}">
                <div class="result-title">{result_icon} Hasil Analisis: {anxiety_result}</div>
                <div class="result-desc"><b>{confidence_text}</b><br>{rec}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Sleep Efficiency Score", f"{sleep_efficiency:.4f}", "Semakin tinggi, pola tidur-kafein semakin baik.", "status-good")
        with c2:
            metric_card("Lifestyle Risk Index", f"{lifestyle_risk:.4f}", "Ringkasan risiko dari stres, kebiasaan, dan riwayat.", "status-mid")
        with c3:
            metric_card("Anxiety Composite Score", f"{anxiety_composite:.4f}", "Skor gabungan indikator psikologis dan fisik.", "status-high")

        st.markdown("---")
        c4, c5 = st.columns([0.9, 1.1])
        with c4:
            score_df = pd.DataFrame(
                {
                    "Indikator": ["Sleep Efficiency", "Lifestyle Risk", "Anxiety Composite"],
                    "Skor": [sleep_efficiency, lifestyle_risk, anxiety_composite],
                }
            )
            fig, ax = plt.subplots(figsize=(7, 4.2))
            sns.barplot(data=score_df, x="Skor", y="Indikator", color=MAROON_SOFT, ax=ax)
            ax.set_xlim(0, 1)
            ax.set_title("Skor Feature Engineering", fontsize=14, fontweight="bold", color=MAROON_DARK)
            ax.grid(axis="x", alpha=.18)
            sns.despine(ax=ax)
            fig_to_streamlit(fig)

        with c5:
            if class_values is not None and len(class_values) >= 3:
                proba_df = pd.DataFrame(
                    {
                        "Kelas": ["Low", "Medium", "High"],
                        "Probabilitas": class_values[:3],
                    }
                )
                fig, ax = plt.subplots(figsize=(7, 4.2))
                sns.barplot(data=proba_df, x="Kelas", y="Probabilitas", palette=[GREEN, GOLD, RED], ax=ax)
                ax.set_ylim(0, max(1, float(np.max(class_values[:3])) + 0.05))
                ax.set_title("Probabilitas Output Model", fontsize=14, fontweight="bold", color=MAROON_DARK)
                ax.grid(axis="y", alpha=.18)
                sns.despine(ax=ax)
                fig_to_streamlit(fig)
            else:
                info_card(
                    "Probabilitas model tidak tersedia",
                    "Grafik probabilitas hanya muncul jika model AI berhasil memberikan output 3 kelas. Karena mode cadangan aktif, dashboard menampilkan skor feature engineering sebagai interpretasi utama.",
                    ["Fallback", "Score-based"],
                )

        st.caption("Catatan: Hasil ini bersifat deteksi awal/simulasi berbasis data dan tidak menggantikan diagnosis tenaga profesional.")


# =========================================================
# PAGE: ABOUT & METHODOLOGY
# =========================================================
def page_about_methodology():
    section_header("ℹ️", "About & Methodology", "Penjelasan singkat mengenai alur kerja dashboard dan model.")

    c1, c2 = st.columns(2)
    with c1:
        info_card(
            "Alur dashboard",
            "Dataset dibaca dari folder data, nama kolom diseragamkan, lalu dipakai untuk EDA. Model .keras dibaca dari folder models dan digunakan hanya untuk inference/prediksi awal.",
            ["Load Data", "Clean Columns", "EDA", "Predict"],
        )
        info_card(
            "Input model",
            "Form prediksi membentuk 21 fitur input, mulai dari umur, gender, pekerjaan, durasi tidur, aktivitas fisik, kafein, alkohol, smoking, riwayat keluarga, stress level, indikator fisik, hingga skor feature engineering.",
            ["21 Features", "Keras", "Inference"],
        )
    with c2:
        info_card(
            "Batasan interpretasi",
            "Dashboard ini tidak dapat menggantikan diagnosis psikolog, psikiater, atau tenaga kesehatan. Hasil prediksi hanya digunakan sebagai edukasi, eksplorasi data, dan deteksi awal berbasis pola data.",
            ["Not Medical Diagnosis", "Educational Use", "Screening"],
        )
        info_card(
            "Struktur deploy",
            "Agar deploy lancar, repo GitHub sebaiknya berisi dashboard_capstone.py, requirements.txt, folder data berisi cleaned_anxiety_data.csv, dan folder models berisi mindbalance_model_new.keras.",
            ["Streamlit Cloud", "Python 3.12", "GitHub"],
        )

    st.markdown("---")
    st.subheader("Struktur folder yang disarankan")
    st.code(
        """
mindbalance/
├── dashboard_capstone.py
├── requirements.txt
├── data/
│   └── cleaned_anxiety_data.csv
└── models/
    └── mindbalance_model_new.keras
        """.strip(),
        language="text",
    )

    st.subheader("Requirements")
    st.code(
        """
streamlit
pandas
matplotlib
seaborn
numpy
tensorflow==2.21.0
        """.strip(),
        language="text",
    )

    with st.expander("Lihat kolom dataset yang terbaca"):
        st.write(df.columns.tolist())


# =========================================================
# ROUTER
# =========================================================
if page == "🏠 Executive Overview":
    page_executive_overview()
elif page == "📊 Dataset Overview":
    page_dataset_overview()
elif page == "🌙 Lifestyle Insights":
    page_lifestyle_insights()
elif page == "🫀 Physical Indicators":
    page_physical_indicators()
elif page == "🔗 Correlation Explorer":
    page_correlation_explorer()
elif page == "🔮 AI Detection Form":
    page_ai_detection_form()
elif page == "ℹ️ About & Methodology":
    page_about_methodology()


# =========================================================
# FOOTER
# =========================================================
st.markdown(
    """
    <div class="footer-box">
        <b>MindBalance Dashboard</b><br>
        Dashboard ini dibuat untuk visualisasi data dan simulasi deteksi awal tingkat kecemasan.
        Hasil prediksi bukan diagnosis medis final. Jika mengalami gejala berat atau mengganggu aktivitas,
        konsultasikan dengan tenaga profesional.
    </div>
    """,
    unsafe_allow_html=True,
)
