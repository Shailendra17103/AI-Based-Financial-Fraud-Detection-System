import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix, classification_report, precision_score,
    recall_score, f1_score, accuracy_score
)

# ---------------------------------------------------------------------
# Load model + artifacts
# ---------------------------------------------------------------------
model = joblib.load("fraud_xgb_model.pkl")

# FIX: Add missing attributes for older XGBoost models to avoid crashes
setattr(model, "use_label_encoder", False)
setattr(model, "gpu_id", None)
setattr(model, "n_gpus", 0)
setattr(model, "predictor", "cpu_predictor")

scaler = joblib.load("fraud_scaler.pkl")

# Optimized threshold (F1-maximizing)
threshold = 0.8676844

# ---------------------------------------------------------------------
# Preprocessing helper
# ---------------------------------------------------------------------
def preprocess(df):
    df = df.copy()

    # Time-based features
    if "hour" not in df.columns and "Time" in df.columns:
        df["hour"] = (df["Time"] // 3600) % 24
    if "day" not in df.columns and "Time" in df.columns:
        df["day"] = (df["Time"] // (24 * 3600))

    # Amount log
    if "amount_log" not in df.columns and "Amount" in df.columns:
        df["amount_log"] = np.log1p(df["Amount"])

    # Scaled amount
    if "amount_log" in df.columns:
        df["amount_log_scaled"] = scaler.transform(df[["amount_log"]])
    else:
        df["amount_log"] = 0.0
        df["amount_log_scaled"] = scaler.transform(df[["amount_log"]])

    return df

# Correct model feature ordering
MODEL_FEATURES = [
    'V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
    'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
    'V21','V22','V23','V24','V25','V26','V27','V28',
    'Amount','hour','day','amount_log','amount_log_scaled'
]

# ---------------------------------------------------------------------
# Streamlit UI - Navigation
# ---------------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:",
                       ["Fraud Prediction",
                        "Visual Analytics",
                        "Model Explainability",
                        "Model Performance",
                        "About Project"])

st.title("💳 AI-Based Financial Fraud Detection System")

# ---------------------------------------------------------------------
# PAGE: FRAUD PREDICTION
# ---------------------------------------------------------------------
if page == "Fraud Prediction":
    st.header("🔍 Predict Fraud from Uploaded CSV")
    uploaded_file = st.file_uploader("Upload CSV File (transactions)", type=["csv"], key="pred")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("Preview (first 5 rows)")
        st.write(df.head())

        df_prep = preprocess(df)

        missing = [c for c in MODEL_FEATURES if c not in df_prep.columns]
        if missing:
            st.error(f"Missing required features for prediction: {missing}")
        else:
            probs = model.predict_proba(df_prep[MODEL_FEATURES])[:, 1]
            preds = (probs >= threshold).astype(int)

            df["fraud_probability"] = probs
            df["fraud_prediction"] = preds

            st.subheader("Prediction Output (first 10 rows)")
            st.write(df.head(10))

            st.subheader("Top Suspicious Transactions")
            st.write(df.sort_values("fraud_probability", ascending=False).head(10))

            csv_out = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Results CSV", csv_out, "fraud_predictions.csv")

# ---------------------------------------------------------------------
# PAGE: VISUAL ANALYTICS
# ---------------------------------------------------------------------
elif page == "Visual Analytics":
    st.header("📊 Fraud Insights Dashboard — Interactive Visuals")
    uploaded_file = st.file_uploader("Upload CSV for visualization", type=["csv"], key="viz")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df = preprocess(df)

        if "Class" not in df.columns:
            st.warning("File does not contain 'Class' column — some visualizations require labels.")

        st.subheader("Fraud Count Overview")
        if "Class" in df.columns:
            counts = df["Class"].value_counts().sort_index()
            labels = ["Non-Fraud", "Fraud"] if len(counts) >= 2 else ["Non-Fraud"]
            fig, ax = plt.subplots()
            counts.plot(kind="pie", autopct="%1.2f%%", labels=labels, ax=ax)
            ax.set_ylabel("")
            st.pyplot(fig)

        # Hour-based fraud
        if "hour" in df.columns and "Class" in df.columns:
            st.subheader("Fraud Rate by Hour (line)")
            fig, ax = plt.subplots(figsize=(10, 3))
            hourly = df.groupby("hour")["Class"].mean()
            hourly.plot(kind="line", marker="o", ax=ax)
            ax.grid(True)
            st.pyplot(fig)

        if "day" in df.columns and "Class" in df.columns:
            st.subheader("Fraud Rate by Day (line)")
            fig, ax = plt.subplots(figsize=(10, 3))
            daily = df.groupby("day")["Class"].mean()
            daily.plot(kind="line", marker="o", ax=ax)
            ax.grid(True)
            st.pyplot(fig)

        if "hour" in df.columns and "day" in df.columns and "Class" in df.columns:
            st.subheader("Fraud Heatmap: Day × Hour")
            heatmap_data = df.pivot_table(values="Class", index="day", columns="hour", aggfunc="mean")
            fig, ax = plt.subplots(figsize=(12, 5))
            sns.heatmap(heatmap_data, cmap="Reds", ax=ax)
            st.pyplot(fig)

        st.subheader("Transaction Amount Distribution")
        fig, ax = plt.subplots(figsize=(10, 3))
        df["Amount"].hist(bins=60, ax=ax)
        st.pyplot(fig)

        if "Class" in df.columns:
            st.subheader("Amount Distribution: Fraud vs Non-Fraud (KDE)")
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.kdeplot(df[df["Class"] == 0]["Amount"], label="Non-Fraud", fill=True, ax=ax)
            if df["Class"].sum() > 0:
                sns.kdeplot(df[df["Class"] == 1]["Amount"], label="Fraud", fill=True, ax=ax)
            ax.legend()
            st.pyplot(fig)

        st.subheader("Amount Boxplot")
        fig, ax = plt.subplots(figsize=(8, 3))
        sns.boxplot(x=df["Amount"], ax=ax)
        st.pyplot(fig)

        st.subheader("Correlation Heatmap")
        corr_cols = [c for c in ["Amount","hour","day","amount_log","amount_log_scaled","Class"] if c in df.columns]
        if len(corr_cols) >= 2:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(df[corr_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        else:
            st.info("Not enough numeric columns for heatmap")

# ---------------------------------------------------------------------
# PAGE: MODEL EXPLAINABILITY
# ---------------------------------------------------------------------
elif page == "Model Explainability":
    st.header("🔎 Model Explainability (Feature Importance, Permutation, PDP)")
    st.write("SHAP disabled for compatibility.")

    st.subheader("XGBoost Feature Importance (Gain)")
    booster = model.get_booster()
    try:
        importance_dict = booster.get_score(importance_type='gain')
    except:
        importance_dict = {}

    if importance_dict:
        importance_df = pd.DataFrame({
            "feature": list(importance_dict.keys()),
            "importance": list(importance_dict.values())
        }).sort_values("importance", ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(importance_df["feature"], importance_df["importance"])
        ax.invert_yaxis()
        st.pyplot(fig)
    else:
        st.info("Could not extract importance")

    st.subheader("Permutation Importance (Requires labeled CSV)")
    perm_file = st.file_uploader("Upload CSV", type=["csv"], key="perm_imp")

    if perm_file:
        df_local = pd.read_csv(perm_file)
        df_local = preprocess(df_local)

        missing = [c for c in MODEL_FEATURES if c not in df_local.columns]
        if missing:
            st.error(f"Missing: {missing}")
        else:
            X_local = df_local[MODEL_FEATURES]
            if "Class" not in df_local.columns:
                st.error("Permutation importance requires Class column")
            else:
                y_local = df_local["Class"]
                st.info("Computing permutation importance...")
                result = permutation_importance(model, X_local, y_local, n_repeats=10, random_state=0)

                perm_df = pd.DataFrame({"feature": MODEL_FEATURES, "importance": result.importances_mean})
                perm_df = perm_df.sort_values("importance", ascending=False)

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(perm_df["feature"][:20], perm_df["importance"][:20])
                ax.invert_yaxis()
                st.pyplot(fig)

    st.subheader("Partial Dependence Plot (PDP)")
    pdp_file = st.file_uploader("Upload CSV for PDP", type=["csv"])

    if pdp_file:
        df_pdp = pd.read_csv(pdp_file)
        df_pdp = preprocess(df_pdp)

        sel_feature = st.selectbox("Select feature", [f for f in MODEL_FEATURES if f in df_pdp.columns])

        if sel_feature:
            X_small = df_pdp[[c for c in MODEL_FEATURES if c in df_pdp.columns]].sample(min(300, len(df_pdp)), random_state=1)
            fig, ax = plt.subplots(figsize=(8, 4))
            try:
                PartialDependenceDisplay.from_estimator(model, X_small, [sel_feature], ax=ax)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"PDP failed: {e}")

# ---------------------------------------------------------------------
# PAGE: MODEL PERFORMANCE
# ---------------------------------------------------------------------
elif page == "Model Performance":
    st.header("📈 Model Performance Dashboard")
    perf_file = st.file_uploader("Upload labeled CSV", type=["csv"], key="perf")

    if perf_file:
        df_eval = pd.read_csv(perf_file)
        if "Class" not in df_eval.columns:
            st.error("Missing Class column")
        else:
            df_eval = preprocess(df_eval)
            missing = [c for c in MODEL_FEATURES if c not in df_eval.columns]

            if missing:
                st.error(f"Missing: {missing}")
            else:
                X = df_eval[MODEL_FEATURES]
                y = df_eval["Class"]

                prob = model.predict_proba(X)[:, 1]
                pred = (prob >= threshold).astype(int)

                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y, pred)
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                st.pyplot(fig)

                st.subheader("ROC Curve")
                fpr, tpr, _ = roc_curve(y, prob)
                auc_score = auc(fpr, tpr)
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
                ax.plot([0, 1], [0, 1], "k--")
                ax.legend()
                st.pyplot(fig)

                st.subheader("Precision–Recall Curve")
                precision, recall, _ = precision_recall_curve(y, prob)
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(recall, precision)
                st.pyplot(fig)

                st.subheader("Threshold Tuning")
                thresholds = np.linspace(0, 1, 200)
                precs, recs, f1s = [], [], []
                for t in thresholds:
                    p = (prob >= t).astype(int)
                    precs.append(precision_score(y, p, zero_division=0))
                    recs.append(recall_score(y, p))
                    f1s.append(f1_score(y, p))

                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(thresholds, precs, label="Precision")
                ax.plot(thresholds, recs, label="Recall")
                ax.plot(thresholds, f1s, label="F1")
                ax.axvline(threshold, color="red", linestyle="--", label=f"Selected threshold = {threshold}")
                ax.legend()
                st.pyplot(fig)

                st.subheader("Classification Report")
                st.text(classification_report(y, pred, zero_division=0))

# ---------------------------------------------------------------------
# PAGE: ABOUT PROJECT
# ---------------------------------------------------------------------
elif page == "About Project":
    st.header("📘 Project Overview")
    st.write("""
    **AI-Based Financial Fraud Detection System**

    - XGBoost classifier trained on anonymized credit card transactions  
    - Feature engineering (time features, log amount, scaled amount)  
    - Threshold optimization (F1-based)  
    - Explainability: XGBoost gain importance, permutation importance, PDP  
    - Visual analytics: histograms, KDE, heatmaps, boxplots

    Developer: **Shailendra Bhushan Rai**
    """)
