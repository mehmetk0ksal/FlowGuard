import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.metrics import accuracy_score, classification_report
from openai import OpenAI
import httpx


# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Cyber Shield IDS", page_icon="🛡️", layout="wide")


# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "final_cyber_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "configs", "selected_features.json")
API_CONFIG_PATH = os.path.join(BASE_DIR, "configs", "api_config.json")


# --- DICTIONARIES ---
CATEGORY_DICT = {
    'Benign': 0, 'Reconnaissance': 1, 'Analysis': 1,
    'Exploits': 2, 'Shellcode': 2, 'Backdoor': 2, 'Worms': 2,
    'DoS': 3, 'Fuzzers': 4, 'Generic': 4
}

REVERSE_DICT = {0: "Normal", 1: "Reconnaissance", 2: "Exploits", 3: "DoS", 4: "Generic"}


# --- CACHED LOADERS ---
@st.cache_resource
def load_system_files():
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, 'r', encoding='utf-8') as f:
        selected_features = json.load(f)
    return model, selected_features


def get_api_key():
    try:
        with open(API_CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)
            return config.get("CEREBRAS_API_KEY", "")
    except FileNotFoundError:
        return ""


def parse_llm_json(response_text):
    """Safely converts text from LLM to JSON (cleans Markdown formatting)."""
    try:
        clean_text = response_text.strip()
        if clean_text.startswith('```json'):
            clean_text = clean_text[7:]
        elif clean_text.startswith('```'):
            clean_text = clean_text[3:]
        if clean_text.endswith('```'):
            clean_text = clean_text[:-3]
        return json.loads(clean_text.strip())
    except Exception:
        return None


# --- SEMANTIC FEATURE TRANSLATOR ---
def build_semantic_summary(row_dict: dict, packet_id: int) -> str:
    """
    Converts a single traffic row into human-readable English sentences
    that an LLM can reason about. Replaces raw numeric feature lists
    with meaningful context — critical for LLM accuracy.
    """
    lines = [f"[Packet {packet_id}]"]

    # Port analysis
    dst = row_dict.get("Dst Port", 0)
    src = row_dict.get("Src Port", 0)
    well_known = {
        80: "HTTP", 443: "HTTPS", 22: "SSH", 21: "FTP",
        23: "Telnet", 25: "SMTP", 445: "SMB", 3389: "RDP",
        53: "DNS", 8080: "HTTP-alt"
    }
    dst_label = well_known.get(int(dst), f"unusual port {int(dst)}")
    lines.append(f"Destination: {dst_label} | Source port: {int(src)}")

    # Traffic intensity
    pps = row_dict.get("Flow Packets/s", 0)
    row_dict.get("Flow Bytes/s", 0)
    if pps > 5000:
        lines.append(f"EXTREMELY HIGH packet rate: {pps:.0f} pkt/s — classic DoS signature.")
    elif pps > 500:
        lines.append(f"High packet rate: {pps:.0f} pkt/s — possible flood.")
    elif pps < 5:
        lines.append(f"Very low packet rate: {pps:.2f} pkt/s — possible slow scan or idle.")
    else:
        lines.append(f"Normal packet rate: {pps:.1f} pkt/s.")

    # Flow duration
    duration_us = row_dict.get("Flow Duration", 0)
    duration_s = duration_us / 1_000_000 if duration_us > 1000 else duration_us
    if duration_s < 0.01:
        lines.append("Flow duration: near-instantaneous — single burst pattern.")
    elif duration_s > 60:
        lines.append(f"Flow duration: {duration_s:.1f}s — long-running session.")
    else:
        lines.append(f"Flow duration: {duration_s:.2f}s.")

    # Packet size
    pkt_max = row_dict.get("Packet Length Max", 0)
    pkt_mean = row_dict.get("Packet Length Mean", row_dict.get("Fwd Packet Length Mean", 0))
    if pkt_max < 60:
        lines.append(f"Tiny packets (max {pkt_max:.0f} B) — scanning or SYN flood indicator.")
    elif pkt_max > 1400:
        lines.append(f"Full-size packets (max {pkt_max:.0f} B, mean {pkt_mean:.0f} B) — bulk data transfer.")
    else:
        lines.append(f"Moderate packet size (max {pkt_max:.0f} B, mean {pkt_mean:.0f} B).")

    # Forward vs backward asymmetry
    fwd_pkts = row_dict.get("Total Fwd Packet", 1)
    bwd_pkts = row_dict.get("Total Bwd packets", 1)
    ratio = row_dict.get("Down/Up Ratio", 0)
    if fwd_pkts > 500 and bwd_pkts < 5:
        lines.append(f"One-sided traffic: {fwd_pkts:.0f} fwd vs {bwd_pkts:.0f} bwd — DoS/flood, no server response.")
    elif fwd_pkts < 3 and bwd_pkts < 3:
        lines.append(f"Minimal bidirectional exchange ({fwd_pkts:.0f}↑ {bwd_pkts:.0f}↓) — probe or single-shot.")
    else:
        lines.append(f"Bidirectional: {fwd_pkts:.0f} fwd / {bwd_pkts:.0f} bwd packets, ratio {ratio:.2f}.")

    # Flag analysis
    fin = row_dict.get("FIN Flag Count", 0)
    psh = row_dict.get("PSH Flag Count", 0)
    if psh > 3:
        lines.append(f"High PSH flag count ({psh:.0f}) — data push, possible exfiltration or exploit payload.")
    if fin == 0 and fwd_pkts > 100:
        lines.append("No FIN flags despite high volume — connection not gracefully closed.")

    # Inter-arrival time (scanning indicator)
    iat_mean = row_dict.get("Flow IAT Mean", 0)
    iat_max = row_dict.get("Flow IAT Max", 0)
    if iat_mean < 100 and pps < 10:
        lines.append(f"Very low IAT ({iat_mean:.1f} µs) with low pps — possible stealth scan.")
    elif iat_max > 1_000_000:
        lines.append(f"Huge IAT max ({iat_max:.0f} µs) — slow/patient attacker pattern.")

    # TCP window size
    fwd_win = row_dict.get("FWD Init Win Bytes", -1)
    bwd_win = row_dict.get("Bwd Init Win Bytes", -1)
    if fwd_win == 0 or bwd_win == 0:
        lines.append("Zero initial window size — SYN scan or malformed handshake.")
    elif fwd_win > 0:
        lines.append(f"TCP window: fwd={int(fwd_win)}, bwd={int(bwd_win)}.")

    return "\n".join(lines)


# --- LLM BATCH SENDER ---
def ask_llm_batch(client, batch_summaries: list) -> list:
    """
    Sends a single batch of semantic summaries to the LLM.
    batch_summaries: [{"packet_id": int, "summary": str}, ...]
    Returns:         [{"packet_id": int, "llm_prediction": str, "reason": str}, ...]
    """
    system_prompt = """You are an expert network security analyst and IDS engine. Your sole task is to classify each network flow into exactly one of these five categories:

CATEGORIES:
- "Normal"         → Legitimate benign traffic (web browsing, file transfers, normal sessions)
- "Reconnaissance" → Port scans, host probes, service enumeration, slow sweeps
- "Exploits"       → Exploit payloads, shellcode injection, backdoor sessions, worm propagation, RDP/SMB abuse
- "DoS"            → Flood attacks, SYN floods, volumetric denial-of-service, zero-response high-volume flows
- "Generic"        → Fuzzing, malformed traffic, anomalous flows not fitting other attack types

KEY DECISION RULES (apply strictly in order):

1. DoS indicators (ALL of these strongly suggest DoS):
   - Flow Packets/s > 1000 AND Total Bwd packets < 10 (one-sided flood)
   - Flow Packets/s > 5000 (volumetric)
   - Tiny packets (Packet Length Max < 100) + very high packet rate → SYN flood = DoS

2. Exploits indicators:
   - Dst Port in {3389 (RDP), 445 (SMB), 21 (FTP), 23 (Telnet), 22 (SSH)} with asymmetric bidirectional traffic
   - High PSH Flag Count (> 5) with moderate packet sizes → payload injection
   - Flow Duration > 100000 µs with few packets and non-standard ports
   - Bwd Packet Length Max >> Fwd Packet Length Max (server sending large response = exploit response)

3. Reconnaissance indicators:
   - Very low Flow Packets/s (< 5) with minimal packets both directions
   - Flow IAT Max > 500000 µs (slow/patient scanning)
   - FWD Init Win Bytes = 0 (SYN scan, no real handshake)
   - Few Total Fwd Packets (< 5) and few Total Bwd packets (< 5)

4. Generic indicators:
   - Unusual ports + random packet sizes + no clear flood or scan pattern
   - High Down/Up Ratio with non-standard destination
   - Bwd Packets/s very high but not matching DoS pattern

5. Normal indicators:
   - Dst Port in {80, 443, 8080} with balanced bidirectional traffic
   - Smooth Flow IAT Mean (100–5000 µs range)
   - Healthy FWD Init Win Bytes (> 1000) and Bwd Init Win Bytes (> 1000)
   - Packet Length Mean in 200–1400 range with bidirectional balance

OUTPUT FORMAT:
- Reply ONLY with valid JSON — absolutely no extra text, no markdown, no explanation outside JSON.
- Include EVERY packet_id from the input — do not skip any.
- Keep "reason" under 12 words.
- Use ONLY the exact category strings listed above.

{"comparisons":[{"packet_id":0,"llm_prediction":"DoS","reason":"Extremely high packet rate, one-sided, tiny packets."}]}"""

    user_prompt = "\n\n".join(item["summary"] for item in batch_summaries)

    response = client.chat.completions.create(
        model="llama3.1-8b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=1200,  # 10 rows × ~90 tokens/row + structured JSON overhead
    )

    raw = response.choices[0].message.content
    parsed = parse_llm_json(raw)

    if parsed and "comparisons" in parsed:
        return parsed["comparisons"]
    else:
        # Fallback: return Unknown for the whole batch if parse fails
        return [
            {
                "packet_id": item["packet_id"],
                "llm_prediction": "Unknown",
                "reason": f"Parse error: {raw[:80]}"
            }
            for item in batch_summaries
        ]


# --- LLM EXPERT FUNCTION ---
def ask_llm_expert(sample_data_json: str, total_rows: int) -> dict:
    """
    Main LLM entry point — same signature as before, no changes needed in UI code.

    Key changes vs old version:
      • Raw JSON → semantic summaries (critical for LLM accuracy)
      • BATCH_SIZE = 10 rows/request  (avoids token/timeout limits)
      • Each batch is a separate API call — no single large request
      • All results are merged and returned as one dict
    """
    BATCH_SIZE = 10  # Lower to 5 if timeouts persist

    api_key = get_api_key()
    if not api_key:
        return {"error": "CEREBRAS_API_KEY not found. Please check configs/api_config.json."}

    try:
        custom_http_client = httpx.Client(timeout=90.0)
        client = OpenAI(
            base_url="https://api.cerebras.ai/v1",
            api_key=api_key,
            http_client=custom_http_client,
        )

        # Step 1: Parse JSON string to list of dicts
        rows: list = json.loads(sample_data_json)

        # Step 2: Convert each row to a semantic summary
        summaries = []
        for i, row in enumerate(rows):
            pid = row.get("packet_id", i)
            summaries.append({
                "packet_id": pid,
                "summary": build_semantic_summary(row, pid),
            })

        # Step 3: Send in batches
        all_comparisons = []
        total_batches = (len(summaries) + BATCH_SIZE - 1) // BATCH_SIZE

        for batch_idx in range(total_batches):
            batch = summaries[batch_idx * BATCH_SIZE: (batch_idx + 1) * BATCH_SIZE]
            print(f"[LLM] Sending batch {batch_idx + 1}/{total_batches} ({len(batch)} rows)...")
            results = ask_llm_batch(client, batch)
            all_comparisons.extend(results)

        return {"comparisons": all_comparisons}

    except Exception as e:
        error_msg = f"Type: {type(e).__name__} | Details: {str(e)}"
        print(f"\n[CRITICAL LLM ERROR] {error_msg}\n")
        return {"error": f"LLM Connection Failed → {error_msg}"}


# --- BACKEND PREPROCESSING ---
def preprocess_demo_data(df, selected_features):
    df_processed = df.copy()

    leakage_columns = ['Src IP', 'Dst IP', 'Timestamp', 'Flow ID']
    existing_leakage = [col for col in leakage_columns if col in df_processed.columns]
    if existing_leakage:
        df_processed.drop(columns=existing_leakage, inplace=True)

    has_labels = False
    if 'Label' in df_processed.columns:
        has_labels = True
        if isinstance(df_processed['Label'].dropna().iloc[0], str):
            df_processed['Label'] = df_processed['Label'].map(CATEGORY_DICT)
        df_processed.dropna(subset=['Label'], inplace=True)
        df_processed['Label'] = df_processed['Label'].astype(int)

    missing_features = [f for f in selected_features if f not in df_processed.columns]
    if missing_features:
        return None, None, f"Missing features: {missing_features}", False

    X = df_processed[selected_features]
    y = df_processed['Label'] if has_labels else None

    for col in ['Dst Port', 'Src Port']:
        if col in X.columns:
            X[col] = X[col].astype('category')

    return X, y, "Success", has_labels


# --- UI ---
def main():
    st.title("🛡️ Network Intrusion Detection System")
    st.markdown("Powered by an ensemble AI architecture (XGBoost + LightGBM).")

    tab1, tab2 = st.tabs(["📊 Model Performance (Dashboard)", "🚀 Live Demo (Upload Traffic)"])

    with tab1:
        st.header("Test Results (Real-World Performance)")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Packets Analyzed", "522,051", "14.82 seconds")
        col2.metric("Latency per Packet", "0.0284 ms", "Real-time Capable")
        col3.metric("Weighted F1-Score", "98.58%")

        st.divider()

        st.subheader("Critical Security Analysis")
        col_sec1, col_sec2 = st.columns(2)

        with col_sec1:
            st.info("**Threat Leakage Rate**")
            st.markdown("### 1.77%")
            st.write("Out of 13,438 real attacks, only 238 bypassed the system.")
            st.success("[STATUS: EXCELLENT] The model is acting as a robust shield.")

        with col_sec2:
            st.info("**Detection Rates by Category (Recall)**")
            st.write("- **Normal Traffic:** 98.74%")
            st.write("- **Exploits:** 91.49%")
            st.write("- **Generic:** 87.19%")
            st.write("- **Reconnaissance:** 85.40%")
            st.write("- **DoS:** 21.49% ⚠️")

    with tab2:
        st.header("Analyze Network Traffic")
        uploaded_file = st.file_uploader("Upload Network Data (e.g., demo_traffic.csv)", type=["csv"])

        if uploaded_file is not None:
            df_demo = pd.read_csv(uploaded_file)
            st.write(f"**Loaded:** {len(df_demo)} rows.")

            try:
                model, selected_features = load_system_files()
            except Exception:
                st.error("Model or Configuration files not found!")
                st.stop()

            with st.spinner("Preprocessing and mapping features..."):
                X_demo, y_demo, msg, has_labels = preprocess_demo_data(df_demo, selected_features)

            if X_demo is None:
                st.error(msg)
                st.stop()

            with st.spinner("AI is analyzing traffic..."):
                predictions = model.predict(X_demo)

            st.success("Analysis Complete!")

            df_results = df_demo.copy()
            df_results['AI_Prediction'] = [REVERSE_DICT.get(pred, "Unknown") for pred in predictions]
            df_results['packet_id'] = df_results.index

            st.subheader("Prediction Distribution")
            dist = df_results['AI_Prediction'].value_counts()
            st.bar_chart(dist)

            ml_accuracy = 0
            if has_labels:
                st.subheader("Demo Accuracy Check")
                ml_accuracy = accuracy_score(y_demo, predictions)
                st.metric("Demo Accuracy (ML Model)", f"{ml_accuracy * 100:.2f}%")

                with st.expander("Show Detailed Report (ML Model)"):
                    st.text(classification_report(y_demo, predictions))

            with st.expander("View Raw Predictions Dataframe"):
                st.dataframe(df_results.head(100))

            # ── LLM SHOWDOWN ──────────────────────────────────────────────────
            st.divider()
            st.subheader("⚔️ AI Showdown: ML Model vs. LLM (Blind Test)")
            st.write(
                "The network data will be sent to the LLM (Llama 3) with labels completely stripped. "
                "The LLM will perform its own independent feature analysis to predict attack types, "
                "just like the ML model — but using natural language reasoning instead of learned weights."
            )

            if st.button("Run Blind Test via LLM 🚀"):
                subset_df = df_results
                subset_X = X_demo

                st.info("⏳ WAITING... Llama 3 is analyzing all rows in batches — should not be long.")
                with st.spinner(
                    f"Llama 3 is analyzing {len(subset_df)} rows in batches of 10. "
                    "This may take 20–40 seconds..."
                ):
                    # Prepare data: features + packet_id, NO labels
                    df_to_llm = subset_X.copy()
                    df_to_llm['packet_id'] = subset_df['packet_id'].values
                    sample_json = df_to_llm.to_json(orient="records")

                    llm_result = ask_llm_expert(sample_json, len(subset_df))

                if "error" in llm_result:
                    st.error(llm_result["error"])
                else:
                    st.success("LLM Analysis Complete!")

                    comp_data = llm_result.get("comparisons", [])
                    if comp_data:
                        comp_df = pd.DataFrame(comp_data)

                        columns_to_merge = ['packet_id', 'AI_Prediction']
                        if has_labels:
                            columns_to_merge.append('Label')

                        showdown_table = pd.merge(
                            subset_df[columns_to_merge], comp_df,
                            on='packet_id', how='left'
                        )

                        llm_answered_count = showdown_table['llm_prediction'].notna().sum()
                        if llm_answered_count < len(subset_df):
                            st.warning(
                                f"Warning: LLM only answered {llm_answered_count} "
                                f"out of {len(subset_df)} rows due to token limits."
                            )
                            showdown_table.dropna(subset=['llm_prediction'], inplace=True)

                        llm_accuracy_text = "N/A"
                        if has_labels:
                            showdown_table['True Label'] = showdown_table['Label'].apply(
                                lambda x: REVERSE_DICT.get(CATEGORY_DICT.get(x, -1), "Unknown")
                            )

                            correct_llm = (showdown_table['True Label'] == showdown_table['llm_prediction']).sum()
                            llm_accuracy = correct_llm / len(showdown_table)
                            llm_accuracy_text = f"{llm_accuracy * 100:.2f}%"

                            ml_subset_correct = (showdown_table['True Label'] == showdown_table['AI_Prediction']).sum()
                            ml_subset_accuracy = ml_subset_correct / len(showdown_table)

                            col_ml, col_llm = st.columns(2)
                            col_ml.metric("🤖 ML Model Accuracy (Subset)", f"{ml_subset_accuracy * 100:.2f}%")
                            col_llm.metric("🧠 LLM Accuracy (Llama 3)", llm_accuracy_text)

                        if has_labels:
                            final_table = showdown_table[
                                ['packet_id', 'True Label', 'AI_Prediction', 'llm_prediction', 'reason']
                            ].copy()
                            final_table.columns = [
                                'Packet ID', 'True Label', 'ML Prediction', 'LLM Prediction', 'LLM Reasoning'
                            ]
                        else:
                            final_table = showdown_table[
                                ['packet_id', 'AI_Prediction', 'llm_prediction', 'reason']
                            ].copy()
                            final_table.columns = [
                                'Packet ID', 'ML Prediction', 'LLM Prediction', 'LLM Reasoning'
                            ]

                        st.dataframe(final_table, use_container_width=True)


if __name__ == "__main__":
    main()