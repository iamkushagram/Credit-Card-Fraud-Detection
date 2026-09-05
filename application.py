import streamlit as st
import numpy as np
import pickle
import pandas as pd

# ── Page Setup ──────────────────────────────────────────────────
st.set_page_config(page_title="Fraud Detector", page_icon="🛡️", layout="wide")

# ── Load Model & Scaler ─────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('pkl/fraud_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('pkl/fraud_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# ── Title ────────────────────────────────────────────────────────
st.title("🛡️ Credit Card Fraud Detector")
st.write("Enter transaction details below to check if it is fraudulent or legitimate.")

# ── Tabs ─────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Single Prediction", "📂 Batch Prediction"])


# ════════════════════════════════════════════════════════════════
# TAB 1 — Single Prediction
# ════════════════════════════════════════════════════════════════
with tab1:

    st.subheader("Transaction Details")

    # ── Threshold Slider ─────────────────────────────────────────
    threshold = st.slider("Detection Threshold", 0.1, 0.9, 0.5, 0.05)

    # ── Time & Amount ─────────────────────────────────────────────
    col1, col2 = st.columns(2)
    time_val   = col1.number_input("Time (seconds)", value=0.0)
    amount_val = col2.number_input("Amount ($)", value=0.0, min_value=0.0)

    # ── V1 to V28 Inputs ─────────────────────────────────────────
    st.write("**PCA Features (V1 to V28)**")

    v = {}
    cols = st.columns(4)
    for i in range(1, 29):
        v[i] = cols[(i - 1) % 4].number_input(f"V{i}", value=0.0, format="%.4f", key=f"v{i}")

    # ── Predict Button ────────────────────────────────────────────
    if st.button("🔍 Check Transaction"):

        # Scale only Amount (scaler was trained on Amount only)
        amount_scaled = scaler.transform([[amount_val]])[0][0]

        # Build feature array with scaled Amount
        features = [time_val] + [v[i] for i in range(1, 29)] + [amount_scaled]
        X_scaled = np.array(features).reshape(1, -1)

        # Predict
        probability = model.predict_proba(X_scaled)[0][1]
        is_fraud    = probability >= threshold

        st.divider()

        # ── Result ────────────────────────────────────────────────
        if is_fraud:
            st.error("🚨 FRAUD DETECTED — This transaction is suspicious!")
        else:
            st.success("✅ LEGITIMATE — This transaction looks genuine.")

        # ── Metrics ───────────────────────────────────────────────
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Fraud Probability", f"{probability * 100:.2f}%")
        col_b.metric("Threshold Used",    f"{threshold:.2f}")
        col_c.metric("Amount",            f"${amount_val:,.2f}")

        # ── Feature Summary Table ─────────────────────────────────
        st.write("**Transaction Summary**")
        summary = pd.DataFrame({
            "Feature": ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)],
            "Value":   [time_val, amount_val] + [v[i] for i in range(1, 29)]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════
# TAB 2 — Batch Prediction
# ════════════════════════════════════════════════════════════════
with tab2:

    st.subheader("Batch Transaction Analysis")
    st.write("Upload a CSV file with columns: Time, V1–V28, Amount")

    # ── Threshold ─────────────────────────────────────────────────
    batch_threshold = st.slider("Detection Threshold", 0.1, 0.9, 0.5, 0.05, key="batch_thresh")

    # ── File Upload ───────────────────────────────────────────────
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        # Remove Class column if present
        if "Class" in df.columns:
            df = df.drop(columns=["Class"])

        st.write("**Preview — First 5 Rows**")
        st.dataframe(df.head(), use_container_width=True)

        if st.button("🚀 Run Batch Prediction"):

            # Scale only Amount column (scaler was trained on Amount only)
            df["Amount"] = scaler.transform(df[["Amount"]])
            probs    = model.predict_proba(df.values)[:, 1]
            preds    = (probs >= batch_threshold).astype(int)

            # Add results to dataframe
            df["Fraud_Probability"] = probs
            df["Result"]            = ["FRAUD" if p == 1 else "LEGIT" for p in preds]

            # ── Summary Metrics ───────────────────────────────────
            total      = len(preds)
            frauds     = int(preds.sum())
            legits     = total - frauds
            fraud_pct  = (frauds / total) * 100

            st.divider()
            st.subheader("Results Summary")

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Transactions", total)
            col2.metric("Fraud Detected",     frauds)
            col3.metric("Legitimate",         legits)

            if frauds > 0:
                st.error(f"🚨 {frauds} fraudulent transactions found ({fraud_pct:.2f}%)")
            else:
                st.success("✅ No fraud detected in this batch!")

            # ── Full Results Table ────────────────────────────────
            st.write("**Full Predictions**")
            st.dataframe(df[["Fraud_Probability", "Result"]].assign(
                Fraud_Probability=df["Fraud_Probability"].map("{:.4f}".format)
            ), use_container_width=True)

            # ── Flagged Transactions ──────────────────────────────
            flagged = df[df["Result"] == "FRAUD"]
            if not flagged.empty:
                st.write(f"**🚨 Flagged Transactions ({len(flagged)})**")
                st.dataframe(flagged, use_container_width=True)