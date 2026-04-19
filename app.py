import os
import re
import glob
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score
)

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="FlowGuard ",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# PATHS
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "final_cyber_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "configs", "selected_features.json")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

CLASS_NAMES = {
    0: "Normal",
    1: "Recon",
    2: "Exploits",
    3: "DoS",
    4: "Generic"
}

# ---------------------------------------------------
# CYBERPUNK CSS
# ---------------------------------------------------
st.markdown("""
<style>
:root {
    --bg: #070b14;
    --panel: rgba(17, 25, 40, 0.72);
    --panel-2: rgba(10, 16, 30, 0.82);
    --cyan: #00f7ff;
    --pink: #ff2bd6;
    --violet: #8b5cf6;
    --text: #e6f7ff;
    --muted: #8aa0b5;
    --success: #00ff9f;
    --warning: #ffcc00;
    --danger: #ff4d6d;
}

html, body, [class*="css"]  {
    font-family: "Segoe UI", sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(0,247,255,0.08), transparent 25%),
        radial-gradient(circle at top right, rgba(255,43,214,0.08), transparent 25%),
        radial-gradient(circle at bottom center, rgba(139,92,246,0.08), transparent 30%),
        linear-gradient(180deg, #040814 0%, #060b17 100%);
    color: var(--text);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(10,16,30,0.95), rgba(7,11,20,0.95));
    border-right: 1px solid rgba(0,247,255,0.18);
}

.block-container {
    padding-top: 1.8rem;
    padding-bottom: 2rem;
}

.cyber-title {
    font-size: 2.3rem;
    font-weight: 800;
    color: #f5fbff;
    text-shadow: 0 0 10px rgba(0,247,255,0.20);
    margin-bottom: 0.2rem;
}

.cyber-subtitle {
    color: #9ab4c9;
    font-size: 0.98rem;
    margin-bottom: 1.2rem;
}

.glass-card {
    background: rgba(15, 23, 42, 0.55);
    border: 1px solid rgba(0,247,255,0.18);
    box-shadow:
        0 0 0.5px rgba(0,247,255,0.5),
        0 0 22px rgba(0,247,255,0.08),
        inset 0 0 18px rgba(255,255,255,0.02);
    border-radius: 18px;
    padding: 1rem 1rem 0.8rem 1rem;
    backdrop-filter: blur(10px);
}

.metric-card {
    background: linear-gradient(135deg, rgba(10,20,35,0.88), rgba(18,30,50,0.72));
    border: 1px solid rgba(255,255,255,0.08);
    border-left: 3px solid var(--cyan);
    border-radius: 16px;
    padding: 1rem;
    min-height: 110px;
    box-shadow: 0 0 18px rgba(0,247,255,0.08);
}

.metric-title {
    color: #8fb3c9;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.4rem;
}

.metric-value {
    color: white;
    font-size: 1.8rem;
    font-weight: 800;
    line-height: 1.1;
}

.metric-delta {
    margin-top: 0.35rem;
    color: #8affc1;
    font-size: 0.85rem;
}

.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #dffcff;
    margin-bottom: 0.8rem;
    text-shadow: 0 0 8px rgba(0,247,255,0.12);
}

.small-muted {
    color: #8ea6b8;
    font-size: 0.9rem;
}

div[data-testid="stFileUploader"] {
    background: rgba(15, 23, 42, 0.50);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 0.5rem;
}

.stButton>button, .stDownloadButton>button {
    background: linear-gradient(90deg, rgba(0,247,255,0.15), rgba(255,43,214,0.15));
    color: white;
    border: 1px solid rgba(0,247,255,0.28);
    border-radius: 12px;
    font-weight: 600;
    box-shadow: 0 0 14px rgba(0,247,255,0.12);
}

.stButton>button:hover, .stDownloadButton>button:hover {
    border: 1px solid rgba(255,43,214,0.40);
    box-shadow: 0 0 18px rgba(255,43,214,0.16);
    color: white;
}

[data-testid="stMetricValue"] {
    color: white;
}

hr {
    border: none;
    border-top: 1px solid rgba(0,247,255,0.14);
    margin: 1rem 0 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HELPERS
# ---------------------------------------------------
@st.cache_resource
def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)

@st.cache_data
def load_selected_features(features_path: str):
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Feature list not found: {features_path}")
    with open(features_path, "r", encoding="utf-8") as f:
        return json.load(f)

def optimize_dtypes_for_trees(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["Dst Port", "Src Port"]:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df

def prepare_input_dataframe(df: pd.DataFrame, selected_features: list[str]) -> pd.DataFrame:
    missing_cols = [col for col in selected_features if col not in df.columns]
    if missing_cols:
        raise ValueError(
            "The CSV file is missing required columns:\n" + ", ".join(missing_cols)
        )
    X = df[selected_features].copy()
    X = optimize_dtypes_for_trees(X)
    return X

def map_prediction_labels(preds):
    return [CLASS_NAMES.get(int(p), f"Unknown-{p}") for p in preds]

def safe_predict_proba(model, X: pd.DataFrame):
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)
        except Exception:
            return None
    return None

def build_results_df(original_df: pd.DataFrame, preds, proba=None):
    result_df = original_df.copy()
    result_df["Predicted_Label_ID"] = preds
    result_df["Predicted_Label_Name"] = map_prediction_labels(preds)

    if proba is not None:
        result_df["Confidence"] = np.max(proba, axis=1)
        for class_id in range(proba.shape[1]):
            class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
            result_df[f"Prob_{class_name}"] = proba[:, class_id]
    return result_df

# ---------------------------------------------------
# LOG PARSER
# ---------------------------------------------------
def find_latest_log_file(logs_dir: str):
    pattern = os.path.join(logs_dir, "final_report_*.txt")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def parse_latest_log(log_path: str):
    if not log_path or not os.path.exists(log_path):
        return None

    with open(log_path, "r", encoding="utf-8") as f:
        text = f.read()

    metrics = {
        "weighted_f1": None,
        "macro_f1": None,
        "total_attacks": None,
        "missed_attacks": None,
        "miss_rate": None,
        "recall_by_class": {}
    }

    # Still matches Turkish log content
    weighted_match = re.search(r"Ağırlıklı - Normal\): %([0-9.]+)", text)
    macro_match = re.search(r"Macro - Zorlayıcı\): %([0-9.]+)", text)
    total_attack_match = re.search(r"Toplam Gerçek Saldırı: (\d+)", text)
    missed_match = re.search(r"İÇERİ SIZDIRDIĞI Saldırı: (\d+) \(Kaçak Oranı: %([0-9.]+)\)", text)

    if weighted_match:
        metrics["weighted_f1"] = float(weighted_match.group(1))
    if macro_match:
        metrics["macro_f1"] = float(macro_match.group(1))
    if total_attack_match:
        metrics["total_attacks"] = int(total_attack_match.group(1))
    if missed_match:
        metrics["missed_attacks"] = int(missed_match.group(1))
        metrics["miss_rate"] = float(missed_match.group(2))

    recall_matches = re.findall(
        r"Tür\s+(\d+)\s+\(([^)]+)\)\s+->\s+%([0-9.]+)\s+başarı",
        text
    )

    for class_id, class_name, recall_val in recall_matches:
        class_id = int(class_id)
        metrics["recall_by_class"][class_id] = {
            "name": class_name.strip(),
            "recall": float(recall_val)
        }

    return metrics

# ---------------------------------------------------
# CHART HELPERS
# ---------------------------------------------------
def make_donut(values, names, title, hole=0.62):
    fig = go.Figure(
        data=[
            go.Pie(
                labels=names,
                values=values,
                hole=hole,
                textinfo="label+percent",
                marker=dict(
                    colors=["#00F7FF", "#FF2BD6", "#8B5CF6", "#00FF9F", "#FFCC00"]
                )
            )
        ]
    )
    fig.update_layout(
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E6F7FF"),
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", y=-0.15)
    )
    return fig

def make_bar(df, x, y, title):
    fig = px.bar(
        df,
        x=x,
        y=y,
        text=y,
        color=y,
        color_continuous_scale=["#8B5CF6", "#00F7FF", "#FF2BD6"]
    )
    fig.update_layout(
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E6F7FF"),
        xaxis_title="",
        yaxis_title="",
        coloraxis_showscale=False
    )
    fig.update_traces(textposition="outside")
    return fig

# ---------------------------------------------------
# LOAD CORE ASSETS
# ---------------------------------------------------
try:
    model = load_model(MODEL_PATH)
    selected_features = load_selected_features(FEATURES_PATH)
except Exception as e:
    st.error(f"Startup error: {e}")
    st.stop()

latest_log_file = find_latest_log_file(LOGS_DIR)
latest_log_metrics = parse_latest_log(latest_log_file) if latest_log_file else None

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ System Node")
    st.markdown("**Model path**")
    st.code(MODEL_PATH, language="text")

    st.markdown("**Feature config**")
    st.code(FEATURES_PATH, language="text")

    st.markdown("**Expected feature count**")
    st.write(len(selected_features))

    st.markdown("---")
    st.markdown("### 📡 Input Protocol")
    st.write(
        "This version expects a **feature-level CSV** directly. "
        "A raw **PCAP** file requires an additional feature extraction layer."
    )

    if latest_log_file:
        st.markdown("---")
        st.markdown("### 🧾 Latest Log")
        st.caption(os.path.basename(latest_log_file))

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.markdown('<div class="cyber-title">🛡️ FlowGuard // Cyber Traffic Analyzer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="cyber-subtitle">Neon-enhanced threat visibility, log analysis, and model-based traffic classification dashboard.</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------
# TOP METRICS FROM LOG
# ---------------------------------------------------
if latest_log_metrics:
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Weighted F1</div>
            <div class="metric-value">%{latest_log_metrics['weighted_f1'] if latest_log_metrics['weighted_f1'] is not None else '-'}</div>
            <div class="metric-delta">Overall field performance</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Macro F1</div>
            <div class="metric-value">%{latest_log_metrics['macro_f1'] if latest_log_metrics['macro_f1'] is not None else '-'}</div>
            <div class="metric-delta">Cross-class balance</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Attacks</div>
            <div class="metric-value">{latest_log_metrics['total_attacks'] if latest_log_metrics['total_attacks'] is not None else '-'}</div>
            <div class="metric-delta">Extracted from test log</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Leak Rate</div>
            <div class="metric-value">%{latest_log_metrics['miss_rate'] if latest_log_metrics['miss_rate'] is not None else '-'}</div>
            <div class="metric-delta">Attacks classified as normal</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Class Recall Distribution</div>', unsafe_allow_html=True)

        recall_data = []
        for class_id, info in latest_log_metrics["recall_by_class"].items():
            recall_data.append({
                "Class": info["name"],
                "Recall": info["recall"]
            })

        if recall_data:
            recall_df = pd.DataFrame(recall_data)
            fig_recall = make_donut(
                values=recall_df["Recall"],
                names=recall_df["Class"],
                title="Recall Donut Chart"
            )
            st.plotly_chart(fig_recall, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with chart_col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Leaked / Blocked Attacks</div>', unsafe_allow_html=True)

        if latest_log_metrics["total_attacks"] is not None and latest_log_metrics["missed_attacks"] is not None:
            caught = latest_log_metrics["total_attacks"] - latest_log_metrics["missed_attacks"]
            fig_leak = make_donut(
                values=[caught, latest_log_metrics["missed_attacks"]],
                names=["Blocked", "Leaked"],
                title="Attack Containment"
            )
            st.plotly_chart(fig_leak, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    recall_bar_col, _ = st.columns([1.2, 0.01])
    with recall_bar_col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Recall Comparison Panel</div>', unsafe_allow_html=True)

        if recall_data:
            recall_df = pd.DataFrame(recall_data)
            fig_bar = make_bar(recall_df, x="Class", y="Recall", title="Class Recall %")
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ---------------------------------------------------
# TABS
# ---------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📂 CSV Prediction", "🧬 Expected Features", "🧾 Log Preview"])

with tab2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Feature List Expected by the Model</div>', unsafe_allow_html=True)
    st.caption(f"Total feature count: {len(selected_features)}")
    st.code("\n".join(selected_features), language="text")
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Latest Final Report Content</div>', unsafe_allow_html=True)
    if latest_log_file:
        with open(latest_log_file, "r", encoding="utf-8") as f:
            st.code(f.read(), language="text")
    else:
        st.warning("No final_report file was found in the logs folder.")
    st.markdown('</div>', unsafe_allow_html=True)

with tab1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">CSV Upload Console</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Select a CSV file", type=["csv"])

    st.markdown(
        '<div class="small-muted">For best results, your CSV should contain the columns expected by the model. '
        'If it also includes a <b>Label</b> column, we can compare against the ground truth.</div>',
        unsafe_allow_html=True
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"CSV uploaded successfully. Rows: {len(df)} | Columns: {len(df.columns)}")
            st.dataframe(df.head(15), use_container_width=True)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        try:
            X = prepare_input_dataframe(df, selected_features)
        except Exception as e:
            st.error(str(e))
            st.stop()

        if st.button("🚀 Start Prediction"):
            with st.spinner("Neural threat matrix engaged..."):
                try:
                    preds = model.predict(X)
                    proba = safe_predict_proba(model, X)
                    results_df = build_results_df(df, preds, proba)
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
                    st.stop()

            st.success("Prediction completed.")

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Records", len(results_df))
            m2.metric("Feature Count", len(selected_features))
            m3.metric("Predicted Class Variety", results_df["Predicted_Label_Name"].nunique())

            dist_df = (
                results_df["Predicted_Label_Name"]
                .value_counts()
                .rename_axis("Class")
                .reset_index(name="Count")
            )

            col_a, col_b = st.columns([1, 1])

            with col_a:
                fig_pred = make_donut(
                    values=dist_df["Count"],
                    names=dist_df["Class"],
                    title="Prediction Distribution"
                )
                st.plotly_chart(fig_pred, use_container_width=True)

            with col_b:
                fig_pred_bar = make_bar(dist_df, x="Class", y="Count", title="Predicted Class Counts")
                st.plotly_chart(fig_pred_bar, use_container_width=True)

            st.markdown("### Results Table")
            st.dataframe(results_df.head(200), use_container_width=True)

            if "Label" in df.columns:
                st.markdown("### Ground Truth Comparison")

                try:
                    y_true = df["Label"]
                    y_pred = preds

                    acc = accuracy_score(y_true, y_pred)
                    macro_f1 = f1_score(y_true, y_pred, average="macro")
                    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

                    a1, a2, a3 = st.columns(3)
                    a1.metric("Accuracy", f"{acc:.4f}")
                    a2.metric("Macro F1", f"{macro_f1:.4f}")
                    a3.metric("Weighted F1", f"{weighted_f1:.4f}")

                    cm = confusion_matrix(y_true, y_pred)
                    cm_df = pd.DataFrame(
                        cm,
                        index=[f"True_{CLASS_NAMES.get(i, i)}" for i in range(cm.shape[0])],
                        columns=[f"Pred_{CLASS_NAMES.get(i, i)}" for i in range(cm.shape[1])]
                    )
                    st.dataframe(cm_df, use_container_width=True)

                    report = classification_report(y_true, y_pred, zero_division=0)
                    st.code(report, language="text")

                except Exception as e:
                    st.warning(f"The Label column was found, but metrics could not be calculated: {e}")

            csv_bytes = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Results CSV",
                data=csv_bytes,
                file_name="flowguard_predictions.csv",
                mime="text/csv"
            )

    st.markdown('</div>', unsafe_allow_html=True)