import os
from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

try:
    import tensorflow as tf
except Exception as e:
    tf = None
    TF_IMPORT_ERROR = e


# =========================================================
# BASE DIRECTORY & FILE FINDER
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


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="MindBalance - Anxiety Detection Dashboard",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 MindBalance — AI-Powered Anxiety Detection & Mental Wellness")
st.markdown("---")


# =========================================================
# LOAD MODEL TENSORFLOW
# =========================================================
@st.cache_resource
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
        recursive_pattern="mindbalance_model_new*.keras"
    )

    if model_path is None:
        raise FileNotFoundError(
            "File model tidak ditemukan. Simpan model di "
            "'models/mindbalance_model_new.keras'."
        )

    # compile=False supaya lebih aman saat model hanya dipakai untuk inference/prediksi
    return tf.keras.models.load_model(str(model_path), compile=False), model_path


try:
    model, model_path_used = load_ai_model()
    model_loaded = True
    st.success(f"✅ Model AI berhasil dimuat: {model_path_used.relative_to(BASE_DIR)}")
except Exception as e:
    st.error(f"❌ Gagal memuat model AI: {e}")
    st.info("Aplikasi tetap dapat berjalan memakai mode cadangan berbasis skor.")
    model = None
    model_loaded = False


# =========================================================
# LOAD & CLEAN DATASET
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
    """Samakan nama kolom dataset agar cocok dengan kode EDA."""
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    alias_map = {
        "Anxiety_Category": [
            "Anxiety Category",
            "Anxiety_Level_Category",
            "Anxiety Level Category",
            "Category"
        ],
        "Sleep_Duration": [
            "Sleep Hours",
            "Sleep_Hours",
            "Sleep Duration",
            "Sleep_Duration_Hours",
            "Sleep Hours Daily"
        ],
        "Caffeine_Intake": [
            "Caffeine Intake (mg/day)",
            "Caffeine Intake",
            "Caffeine_Intake_mg_day",
            "Caffeine"
        ],
        "Stress_Level": [
            "Stress Level (1-10)",
            "Stress Level",
            "Stress_Level_1_10"
        ],
        "Heart_Rate": [
            "Heart Rate (bpm)",
            "Heart Rate",
            "Heart_Rate_bpm"
        ],
        "Breathing_Rate": [
            "Breathing Rate (breaths/min)",
            "Breathing Rate",
            "Breathing_Rate_breaths_min"
        ],
        "Sweating_Level": [
            "Sweating Level (1-5)",
            "Sweating Level",
            "Sweating_Level_1_5"
        ],
        "Physical_Activity": [
            "Physical Activity (hrs/week)",
            "Physical Activity",
            "Physical_Activity_hrs_week"
        ],
        "Alcohol_Consumption": [
            "Alcohol Consumption (drinks/week)",
            "Alcohol Consumption",
            "Alcohol"
        ],
        "Diet_Quality": [
            "Diet Quality (1-10)",
            "Diet Quality",
            "Diet_Quality_1_10"
        ],
        "Anxiety_Level": [
            "Anxiety Level (1-10)",
            "Anxiety Level",
            "Anxiety_Score"
        ],
    }

    for canonical, aliases in alias_map.items():
        df = ensure_column(df, canonical, aliases)

    numeric_cols = [
        "Age", "Sleep_Duration", "Caffeine_Intake", "Stress_Level",
        "Heart_Rate", "Breathing_Rate", "Sweating_Level",
        "Physical_Activity", "Alcohol_Consumption", "Diet_Quality",
        "Anxiety_Level"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Anxiety_Category" in df.columns:
        df["Anxiety_Category"] = df["Anxiety_Category"].astype(str).str.strip()

    return df


@st.cache_data
def load_data():
    data_path = find_first_existing_file(
        candidates=[
            BASE_DIR / "data" / "cleaned_anxiety_data.csv",
            BASE_DIR / "cleaned_anxiety_data.csv",
        ],
        recursive_pattern="cleaned_anxiety_data*.csv"
    )

    if data_path is None:
        raise FileNotFoundError(
            "File dataset tidak ditemukan. Simpan dataset di "
            "'data/cleaned_anxiety_data.csv'."
        )

    df = pd.read_csv(data_path)
    df = prepare_dataset(df)
    return df, data_path


try:
    df, data_path_used = load_data()
    st.success(f"✅ Dataset berhasil dimuat: {data_path_used.relative_to(BASE_DIR)}")
except Exception as e:
    st.warning(f"Dataset tidak ditemukan atau gagal dibaca: {e}")
    st.info("EDA akan memakai data dummy sementara.")

    df = pd.DataFrame({
        "Anxiety_Category": ["Low", "Medium", "High", "Low", "High", "Medium"],
        "Sleep_Duration": [7.5, 6.0, 5.0, 8.0, 4.5, 6.5],
        "Caffeine_Intake": [100, 250, 400, 50, 450, 200],
        "Stress_Level": [3, 6, 9, 2, 8, 5],
        "Heart_Rate": [70, 82, 95, 68, 90, 78]
    })


# =========================================================
# SIDEBAR MENU
# =========================================================
menu = st.sidebar.selectbox(
    "Pilih Menu:",
    ["📊 Interactive EDA", "🔮 Anxiety Detection (Inference UI)"]
)


# =========================================================
# MENU 1: INTERACTIVE EDA
# =========================================================
if menu == "📊 Interactive EDA":
    st.header("Exploratory Data Analysis (EDA) Interaktif")
    st.write("Analisis hubungan antara gaya hidup, kondisi fisik, dan tingkat kecemasan pengguna.")

    required_cols = ["Anxiety_Category", "Sleep_Duration"]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Kolom wajib untuk EDA belum ditemukan: {missing_cols}")
        st.write("Kolom yang tersedia di dataset:", df.columns.tolist())
        st.stop()

    category_order = ["Low", "Medium", "High"]
    available_categories = [
        cat for cat in category_order
        if cat in df["Anxiety_Category"].dropna().unique().tolist()
    ]

    if not available_categories:
        available_categories = sorted(df["Anxiety_Category"].dropna().unique().tolist())

    selected_anxiety = st.multiselect(
        "Filter berdasarkan Tingkat Kecemasan:",
        options=available_categories,
        default=available_categories
    )

    filtered_df = df[df["Anxiety_Category"].isin(selected_anxiety)].copy()

    if filtered_df.empty:
        st.warning("Tidak ada data sesuai filter yang dipilih.")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Distribusi Tingkat Kecemasan")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(
            data=filtered_df,
            x="Anxiety_Category",
            order=[cat for cat in category_order if cat in filtered_df["Anxiety_Category"].unique()],
            palette="Set2",
            ax=ax
        )
        ax.set_xlabel("Tingkat Kecemasan")
        ax.set_ylabel("Jumlah Pengguna")
        st.pyplot(fig)
        st.caption("Grafik ini menunjukkan proporsi data pengguna berdasarkan tingkat kecemasan yang dipilih.")

    with col2:
        st.subheader("2. Hubungan Durasi Tidur & Kecemasan")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(
            data=filtered_df,
            x="Anxiety_Category",
            y="Sleep_Duration",
            order=[cat for cat in category_order if cat in filtered_df["Anxiety_Category"].unique()],
            palette="Pastel1",
            ax=ax
        )
        ax.set_xlabel("Tingkat Kecemasan")
        ax.set_ylabel("Durasi Tidur (Jam)")
        st.pyplot(fig)
        st.caption("Grafik ini menunjukkan perbedaan durasi tidur pada tiap kategori kecemasan.")

    st.markdown("---")
    st.subheader("3. Matriks Korelasi Fitur Numerik")

    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(
            numeric_df.corr(),
            annot=True,
            cmap="Coolwarm",
            fmt=".2f",
            ax=ax
        )
        st.pyplot(fig)
    else:
        st.warning("Kolom numerik belum cukup untuk membuat matriks korelasi.")

    with st.expander("Lihat preview data"):
        st.write("Kolom dataset:", df.columns.tolist())
        st.dataframe(df.head(20))


# =========================================================
# MENU 2: INFERENCE UI
# =========================================================
elif menu == "🔮 Anxiety Detection (Inference UI)":
    st.header("Form Deteksi Dini Tingkat Kecemasan")
    st.write("Masukkan indikator profil klinis dan kebiasaan harian Anda untuk dianalisis oleh AI.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 👤 Profil Demografi & Kebiasaan")

            age = st.number_input("Umur Anda:", min_value=18, max_value=100, value=25)
            gender = st.selectbox("Jenis Kelamin:", ["Male", "Female", "Other"])

            occupation = st.selectbox(
                "Pekerjaan:",
                [
                    "Artist", "Athlete", "Chef", "Doctor", "Engineer",
                    "Freelancer", "Lawyer", "Musician", "Nurse", "Other",
                    "Scientist", "Student", "Teacher"
                ]
            )

            sleep_hours = st.slider("Durasi Tidur Harian (Jam):", 2.3, 11.3, 7.0, 0.1)
            physical_activity = st.slider("Aktivitas Fisik (Jam/Minggu):", 0.0, 10.1, 3.0, 0.1)
            caffeine = st.number_input("Konsumsi Kafein Harian (mg):", min_value=0, max_value=599, value=150)
            alcohol = st.number_input("Konsumsi Alkohol (Gelas/Minggu):", min_value=0, max_value=19, value=2)
            smoking = st.selectbox("Apakah Anda Merokok?", ["Yes", "No"])
            diet_quality = st.slider("Kualitas Pola Makan (Skala 1-10):", 1, 10, 7)

        with col2:
            st.markdown("### 🫀 Parameter Fisik & Medis")

            stress_level = st.slider("Tingkat Stres Psikologis (Skala 1-10):", 1, 10, 5)
            heart_rate = st.number_input("Detak Jantung Istirahat (BPM):", min_value=60, max_value=119, value=75)
            breathing_rate = st.number_input("Frekuensi Napas (Napas/Menit):", min_value=12, max_value=29, value=18)
            sweating_level = st.slider("Tingkat Keringat Berlebih (Skala 1-5):", 1, 5, 2)
            dizziness = st.selectbox("Apakah Sering Merasa Pusing?", ["Yes", "No"])
            medication = st.selectbox("Sedang Konsumsi Obat Kecemasan?", ["Yes", "No"])
            therapy_sessions = st.number_input("Sesi Terapi / Konseling Bulan Ini:", min_value=0, max_value=12, value=0)
            family_history = st.selectbox("Ada Riwayat Kecemasan di Keluarga?", ["Yes", "No"])
            recent_life_event = st.selectbox("Ada Kejadian Besar/Trauma 6 Bulan Terakhir?", ["Yes", "No"])

        submit_button = st.form_submit_button(label="Mulai Deteksi AI")

    if submit_button:
        st.markdown("---")
        st.subheader("📋 Hasil Analisis")

        sleep_norm = (sleep_hours - 2.3) / (11.3 - 2.3)
        caffeine_norm = caffeine / 599.0
        sleep_efficiency = round((sleep_norm * 0.7 + (1 - caffeine_norm) * 0.3), 4)

        binary_map = {"Yes": 1, "No": 0}

        stress_norm = (stress_level - 1) / 9
        alcohol_norm = alcohol / 19.0
        activity_norm = physical_activity / 10.1
        diet_norm = (diet_quality - 1) / 9

        lifestyle_risk = round((
            stress_norm * 0.25 +
            binary_map[smoking] * 0.10 +
            alcohol_norm * 0.10 +
            binary_map[family_history] * 0.15 +
            binary_map[recent_life_event] * 0.15 +
            (1 - activity_norm) * 0.10 +
            (1 - diet_norm) * 0.15
        ), 4)

        hr_norm = (heart_rate - 60) / (119 - 60)
        br_norm = (breathing_rate - 12) / (29 - 12)
        sweat_norm = (sweating_level - 1) / 4

        anxiety_composite = round((
            stress_norm * 0.30 +
            hr_norm * 0.20 +
            br_norm * 0.20 +
            sweat_norm * 0.15 +
            binary_map[family_history] * 0.15
        ), 4)

        if model_loaded:
            gender_map = {"Male": 0, "Female": 1, "Other": 2}

            occ_list = [
                "Artist", "Athlete", "Chef", "Doctor", "Engineer",
                "Freelancer", "Lawyer", "Musician", "Nurse", "Other",
                "Scientist", "Student", "Teacher"
            ]

            occ_map = {occ: i for i, occ in enumerate(occ_list)}

            input_vector = np.array([[
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
                anxiety_composite
            ]], dtype=np.float32)

            try:
                with st.spinner("Model AI sedang memproses matriks klinis Anda..."):
                    outputs = model(tf.constant(input_vector), training=False)

                    if isinstance(outputs, (list, tuple)):
                        class_out = outputs[0]
                    elif isinstance(outputs, dict):
                        class_out = outputs.get("class_output", list(outputs.values())[0])
                    else:
                        class_out = outputs

                    class_values = class_out.numpy()[0] if hasattr(class_out, "numpy") else np.array(class_out)[0]
                    predicted_class_idx = int(np.argmax(class_values))

                labels = ["Low (Rendah)", "Medium (Sedang)", "High (Tinggi)"]
                anxiety_result = labels[predicted_class_idx]

            except Exception as e:
                st.warning(f"Model AI gagal melakukan prediksi, memakai mode cadangan. Detail: {e}")

                if anxiety_composite > 0.5 or stress_level >= 8:
                    predicted_class_idx = 2
                    anxiety_result = "High (Tinggi)"
                elif anxiety_composite > 0.3 or stress_level >= 5:
                    predicted_class_idx = 1
                    anxiety_result = "Medium (Sedang)"
                else:
                    predicted_class_idx = 0
                    anxiety_result = "Low (Rendah)"

        else:
            if anxiety_composite > 0.5 or stress_level >= 8:
                predicted_class_idx = 2
                anxiety_result = "High (Tinggi)"
            elif anxiety_composite > 0.3 or stress_level >= 5:
                predicted_class_idx = 1
                anxiety_result = "Medium (Sedang)"
            else:
                predicted_class_idx = 0
                anxiety_result = "Low (Rendah)"

        if predicted_class_idx == 2:
            st.error(f"**Hasil Analisis Tingkat Kecemasan: {anxiety_result}**")
            st.info(
                "**Rekomendasi Coping:** Ambil waktu jeda istirahat, batasi asupan kopi/kafein harian, "
                "lakukan teknik *box breathing*, dan sangat disarankan untuk berdiskusi dengan psikolog "
                "atau tenaga profesional."
            )

        elif predicted_class_idx == 1:
            st.warning(f"**Hasil Analisis Tingkat Kecemasan: {anxiety_result}**")
            st.info(
                "**Rekomendasi Coping:** Tingkatkan durasi tidur harian, luangkan waktu 15 menit untuk "
                "jalan santai/olahraga, serta kurangi stimulan di malam hari."
            )

        else:
            st.success(f"**Hasil Analisis Tingkat Kecemasan: {anxiety_result}**")
            st.info(
                "**Rekomendasi Coping:** Tingkat kecemasan Anda sangat baik dan stabil. Pertahankan "
                "kombinasi pola hidup dan manajemen stres yang sudah Anda miliki saat ini."
            )

        st.markdown("### Skor Indikator Gabungan (Feature Engineering):")

        m1, m2, m3 = st.columns(3)

        m1.metric("Sleep Efficiency Score", f"{sleep_efficiency:.4f}")
        m2.metric("Lifestyle Risk Index", f"{lifestyle_risk:.4f}")
        m3.metric("Anxiety Composite Score", f"{anxiety_composite:.4f}")

        st.caption(
            "Catatan: Hasil ini bersifat deteksi awal/simulasi berbasis data dan tidak menggantikan diagnosis tenaga profesional."
        )
