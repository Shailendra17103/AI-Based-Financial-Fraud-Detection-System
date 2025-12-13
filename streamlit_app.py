import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)

# =====================================================
# Load model + artifacts
# =====================================================
model = joblib.load("fraud_xgb_model.pkl")

# Compatibility fixes for older XGBoost pickles
setattr(model, "use_label_encoder", False)
setattr(model, "gpu_id", None)
setattr(model, "n_gpus", 0)
setattr(model, "predictor", "cpu_predictor")

scaler = joblib.load("fraud_scaler.pkl")

# Optimized threshold (F1-maximizing)
threshold = 0.8676844

# =====================================================
# Preprocessing
# =====================================================
def preprocess(df):
    df = df.copy()

    if "hour" not in df.columns and "Time" in df.columns:
        df["hour"] = (df["Time"] // 3600) % 24

    if "day" not in df.columns and "Time" in df.columns:
        df["day"] = df["Time"] // (24 * 3600)

    if "amount_log" not in df.columns and "Amount" in df.columns:
        df["amount_log"] = np.log1p(df["Amount"])

    if "amount_log" in df.columns:
        df["amount_log_scaled"] = scaler.transform(df[["amount_log"]])
    else:
        df["amount_log"] = 0.0
        df["amount_log_scaled"] = scaler.transform(df[["amount_log"]])

    return df


MODEL_FEATURES = [
    'V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
    'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
    'V21','V22','V23','V24','V25','V26','V27','V28',
    'Amount','hour','day','amount_log','amount_log_scaled'
]

# =====================================================
# Streamlit UI
# =====================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    [
        "Fraud Prediction",
        "Visual Analytics",
        "Model Explainability",
        "Model Performance",
        "About Project"
    ]
)

st.title("💳 AI-Based Financial Fraud Detection System")

# =====================================================
# FRAUD PREDICTION
# =====================================================
if page == "Fraud Prediction":
    st.header("🔍 Predict Fraud from Uploaded CSV")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write(df.head())

        df_prep = preprocess(df)
        missing = [c for c in MODEL_FEATURES if c not in df_prep.columns]

        if missing:
            st.error(f"Missing required features: {missing}")
        else:
            prob = model.predict_proba(df_prep[MODEL_FEATURES])[:, 1]
            pred = (prob >= threshold).astype(int)

            df["fraud_probability"] = prob
            df["fraud_prediction"] = pred

            st.subheader("Top Suspicious Transactions")
            st.write(df.sort_values("fraud_probability", ascending=False).head(10))

            st.download_button(
                "Download Predictions",
                df.to_csv(index=False).encode("utf-8"),
                "fraud_predictions.csv"
            )

# =====================================================
# VISUAL ANALYTICS
# =====================================================
elif page == "Visual Analytics":
    st.header("📊 Visual Analytics")
    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = preprocess(pd.read_csv(file))

        if "Class" not in df.columns:
            st.warning("Class column missing. Some plots disabled.")

        if "Class" in df.columns:
            st.subheader("Fraud Distribution")
            fig, ax = plt.subplots()
            df["Class"].value_counts().plot(kind="pie", autopct="%1.2f%%", ax=ax)
            ax.set_ylabel("")
            st.pyplot(fig)

        st.subheader("Transaction Amount Distribution")
        fig, ax = plt.subplots()
        df["Amount"].hist(bins=50, ax=ax)
        st.pyplot(fig)

# =====================================================
# MODEL EXPLAINABILITY
# =====================================================
elif page == "Model Explainability":
    st.header("🔎 Model Explainability")

    # -----------------------------
    # XGBoost Feature Importance
    # -----------------------------
    st.subheader("XGBoost Feature Importance (Gain)")
    booster = model.get_booster()
    imp = booster.get_score(importance_type="gain")

    if imp:
        imp_df = (
            pd.DataFrame({"feature": imp.keys(), "importance": imp.values()})
            .sort_values("importance", ascending=False)
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(imp_df["feature"], imp_df["importance"])
        ax.invert_yaxis()
        st.pyplot(fig)
    else:
        st.info("Importance unavailable.")

    # -----------------------------
    # Permutation Importance
    # -----------------------------
    st.subheader("Permutation Importance")
    perm_file = st.file_uploader("Upload labeled CSV", type=["csv"], key="perm")

    if perm_file:
        dfp = preprocess(pd.read_csv(perm_file))

        if "Class" not in dfp.columns:
            st.error("Class column required.")
        else:
            X = dfp[MODEL_FEATURES]
            y = dfp["Class"]

            result = permutation_importance(
                model, X, y, n_repeats=10, random_state=0
            )

            perm_df = (
                pd.DataFrame({
                    "feature": MODEL_FEATURES,
                    "importance": result.importances_mean
                })
                .sort_values("importance", ascending=False)
            )

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(perm_df["feature"][:20], perm_df["importance"][:20])
            ax.invert_yaxis()
            st.pyplot(fig)

    # -----------------------------
    # SAFE MANUAL PDP
    # -----------------------------
    st.subheader("Partial Dependence Plot (Manual, Stable)")
    pdp_file = st.file_uploader("Upload CSV for PDP", type=["csv"], key="pdp")

    if pdp_file:
        df_pdp = preprocess(pd.read_csv(pdp_file))
        features = [f for f in MODEL_FEATURES if f in df_pdp.columns]

        if features:
            feature = st.selectbox("Select feature", features)

            X_base = df_pdp[features].sample(
                min(300, len(df_pdp)),
                random_state=42
            ).copy()

            grid = np.linspace(
                X_base[feature].quantile(0.05),
                X_base[feature].quantile(0.95),
                20
            )

            pdp_vals = []
            for val in grid:
                X_temp = X_base.copy()
                X_temp[feature] = val
                pdp_vals.append(
                    model.predict_proba(X_temp)[:, 1].mean()
                )

            fig, ax = plt.subplots()
            ax.plot(grid, pdp_vals, marker="o")
            ax.set_xlabel(feature)
            ax.set_ylabel("Avg fraud probability")
            ax.grid(True)
            st.pyplot(fig)
        else:
            st.error("No usable features found.")

# =====================================================
# MODEL PERFORMANCE
# =====================================================
elif page == "Model Performance":
    st.header("📈 Model Performance")
    file = st.file_uploader("Upload labeled CSV", type=["csv"])

    if file:
        df = preprocess(pd.read_csv(file))

        if "Class" not in df.columns:
            st.error("Class column missing.")
        else:
            X = df[MODEL_FEATURES]
            y = df["Class"]

            prob = model.predict_proba(X)[:, 1]
            pred = (prob >= threshold).astype(int)

            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y, pred), annot=True, fmt="d", ax=ax)
            st.pyplot(fig)

            st.subheader("Classification Report")
            st.text(classification_report(y, pred, zero_division=0))

# =====================================================
# ABOUT
# =====================================================
elif page == "About Project":
    st.header("📘 About")
    st.write("""
    **AI-Based Financial Fraud Detection System**

    - XGBoost classifier
    - Feature engineering on transaction time & amount
    - Threshold optimization using F1 score
    - Explainability via gain, permutation importance, manual PDP
    - Production-ready Streamlit deployment

    **Developer:** Shailendra Bhushan Rai
    """)
