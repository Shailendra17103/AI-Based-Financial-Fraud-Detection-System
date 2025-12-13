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

# XGBoost compatibility fixes
setattr(model, "use_label_encoder", False)
setattr(model, "gpu_id", None)
setattr(model, "n_gpus", 0)
setattr(model, "predictor", "cpu_predictor")

scaler = joblib.load("fraud_scaler.pkl")

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
# UI
# =====================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Fraud Prediction", "Visual Analytics", "Model Explainability",
     "Model Performance", "About Project"]
)

st.title("💳 AI-Based Financial Fraud Detection System")

# =====================================================
# FRAUD PREDICTION
# =====================================================
if page == "Fraud Prediction":
    st.header("🔍 Fraud Prediction")
    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        df_prep = preprocess(df)

        missing = [c for c in MODEL_FEATURES if c not in df_prep.columns]
        if missing:
            st.error(f"Missing features: {missing}")
        else:
            prob = model.predict_proba(df_prep[MODEL_FEATURES])[:, 1]
            pred = (prob >= threshold).astype(int)

            df["fraud_probability"] = prob
            df["fraud_prediction"] = pred

            st.write(df.head(10))
            st.write(df.sort_values("fraud_probability", ascending=False).head(10))

            st.download_button(
                "Download Predictions",
                df.to_csv(index=False).encode("utf-8"),
                "fraud_predictions.csv"
            )

# =====================================================
# VISUAL ANALYTICS (RESTORED)
# =====================================================
elif page == "Visual Analytics":
    st.header("📊 Visual Analytics")
    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = preprocess(pd.read_csv(file))

        if "Class" not in df.columns:
            st.warning("Class column missing.")

        if "Class" in df.columns:
            st.subheader("Fraud Distribution")
            fig, ax = plt.subplots()
            df["Class"].value_counts().plot(kind="pie", autopct="%1.2f%%", ax=ax)
            ax.set_ylabel("")
            st.pyplot(fig)

            st.subheader("Fraud Rate by Hour")
            fig, ax = plt.subplots()
            df.groupby("hour")["Class"].mean().plot(marker="o", ax=ax)
            ax.set_ylabel("Fraud Rate")
            st.pyplot(fig)

            st.subheader("Fraud Rate by Day")
            fig, ax = plt.subplots()
            df.groupby("day")["Class"].mean().plot(marker="o", ax=ax)
            ax.set_ylabel("Fraud Rate")
            st.pyplot(fig)

            st.subheader("Fraud Heatmap (Day × Hour)")
            heat = df.pivot_table(values="Class", index="day", columns="hour", aggfunc="mean")
            fig, ax = plt.subplots(figsize=(10,4))
            sns.heatmap(heat, cmap="Reds", ax=ax)
            st.pyplot(fig)

        st.subheader("Amount Histogram")
        fig, ax = plt.subplots()
        df["Amount"].hist(bins=60, ax=ax)
        st.pyplot(fig)

        if "Class" in df.columns:
            st.subheader("Amount KDE (Fraud vs Non-Fraud)")
            fig, ax = plt.subplots()
            sns.kdeplot(df[df["Class"]==0]["Amount"], fill=True, label="Non-Fraud", ax=ax)
            sns.kdeplot(df[df["Class"]==1]["Amount"], fill=True, label="Fraud", ax=ax)
            ax.legend()
            st.pyplot(fig)

        st.subheader("Amount Boxplot")
        fig, ax = plt.subplots()
        sns.boxplot(x=df["Amount"], ax=ax)
        st.pyplot(fig)

        st.subheader("Correlation Heatmap")
        cols = [c for c in ["Amount","hour","day","amount_log","amount_log_scaled","Class"] if c in df.columns]
        fig, ax = plt.subplots()
        sns.heatmap(df[cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# =====================================================
# MODEL EXPLAINABILITY (SAFE PDP)
# =====================================================
elif page == "Model Explainability":
    st.header("🔎 Model Explainability")

    st.subheader("XGBoost Feature Importance")
    imp = model.get_booster().get_score(importance_type="gain")
    imp_df = pd.DataFrame({"feature": imp.keys(), "importance": imp.values()}).sort_values("importance", ascending=False)

    fig, ax = plt.subplots()
    ax.barh(imp_df["feature"], imp_df["importance"])
    ax.invert_yaxis()
    st.pyplot(fig)

    st.subheader("Permutation Importance")
    file = st.file_uploader("Upload labeled CSV", type=["csv"], key="perm")
    if file:
        dfp = preprocess(pd.read_csv(file))
        X = dfp[MODEL_FEATURES]
        y = dfp["Class"]

        result = permutation_importance(model, X, y, n_repeats=10, random_state=0)
        perm_df = pd.DataFrame({"feature": MODEL_FEATURES, "importance": result.importances_mean})

        fig, ax = plt.subplots()
        perm_df.sort_values("importance", ascending=False).head(20).plot.barh(x="feature", ax=ax)
        ax.invert_yaxis()
        st.pyplot(fig)

    st.subheader("Partial Dependence Plot (Stable)")
    file = st.file_uploader("Upload CSV for PDP", type=["csv"], key="pdp")
    if file:
        dfp = preprocess(pd.read_csv(file))
        feature = st.selectbox("Select feature", MODEL_FEATURES)

        Xb = dfp[MODEL_FEATURES].sample(min(300,len(dfp)), random_state=42)
        grid = np.linspace(Xb[feature].quantile(0.05), Xb[feature].quantile(0.95), 20)

        pdp = []
        for g in grid:
            Xt = Xb.copy()
            Xt[feature] = g
            pdp.append(model.predict_proba(Xt)[:,1].mean())

        fig, ax = plt.subplots()
        ax.plot(grid, pdp, marker="o")
        ax.set_xlabel(feature)
        ax.set_ylabel("Avg Fraud Probability")
        st.pyplot(fig)

# =====================================================
# MODEL PERFORMANCE (RESTORED)
# =====================================================
elif page == "Model Performance":
    st.header("📈 Model Performance")
    file = st.file_uploader("Upload labeled CSV", type=["csv"])

    if file:
        df = preprocess(pd.read_csv(file))
        X = df[MODEL_FEATURES]
        y = df["Class"]

        prob = model.predict_proba(X)[:,1]
        pred = (prob >= threshold).astype(int)

        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y, pred), annot=True, fmt="d", ax=ax)
        st.pyplot(fig)

        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y, prob)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC={auc(fpr,tpr):.4f}")
        ax.plot([0,1],[0,1],"k--")
        ax.legend()
        st.pyplot(fig)

        st.subheader("Precision-Recall Curve")
        p, r, _ = precision_recall_curve(y, prob)
        fig, ax = plt.subplots()
        ax.plot(r, p)
        st.pyplot(fig)

        st.subheader("Threshold Tuning")
        ts = np.linspace(0,1,200)
        f1s = [f1_score(y, (prob>=t).astype(int)) for t in ts]
        fig, ax = plt.subplots()
        ax.plot(ts, f1s)
        ax.axvline(threshold, color="red", linestyle="--")
        st.pyplot(fig)

        st.subheader("Classification Report")
        st.text(classification_report(y, pred, zero_division=0))

# =====================================================
# ABOUT
# =====================================================
elif page == "About Project":
    st.write("""
    **AI-Based Financial Fraud Detection System**

    Production-grade fraud detection with explainability,
    threshold tuning, and interactive analytics.

    **Developer:** Shailendra Bhushan Rai
    """)
