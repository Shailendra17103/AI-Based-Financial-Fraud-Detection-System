import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved artifacts
model = joblib.load("fraud_xgb_model.pkl")
scaler = joblib.load("fraud_scaler.pkl")

with open("fraud_threshold.txt", "r") as f:
    threshold = float(f.read().strip())

st.title("💳 AI-Based Financial Fraud Detection System")
st.write("Upload a CSV containing transaction data to detect potential fraudulent transactions.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # --- PREPROCESSING ---
    data["hour"] = (data["Time"] // 3600) % 24
    data["day"] = (data["Time"] // (3600 * 24))
    data["amount_log"] = np.log1p(data["Amount"])
    data["amount_log_scaled"] = scaler.transform(data[['amount_log']])

    # Required feature order (MUST match training)
    model_features = [
        'V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
        'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
        'V21','V22','V23','V24','V25','V26','V27','V28',
        'Amount','hour','day','amount_log','amount_log_scaled'
    ]

    # Check for missing columns
    missing = [col for col in model_features if col not in data.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
    else:
        # --- PREDICTION ---
        probs = model.predict_proba(data[model_features])[:, 1]
        preds = (probs >= threshold).astype(int)

        data["fraud_probability"] = probs
        data["fraud_prediction"] = preds

        st.subheader("🔍 Results (First 10 rows)")
        st.write(data.head(10))

        st.subheader("🚨 Top Suspicious Transactions")
        st.write(data.sort_values("fraud_probability", ascending=False).head(10))
