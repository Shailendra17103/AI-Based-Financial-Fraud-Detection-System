# 💳 AI-Based Financial Fraud Detection System

An end-to-end machine learning–powered fraud detection system built using XGBoost, advanced feature engineering, threshold optimization, explainability techniques, and an interactive Streamlit web application.

This project is designed to be industry-ready, deployment-safe, and interview-defensible.

---

## 🚀 Project Highlights

✅ High-performance XGBoost classifier for fraud detection  
✅ Handles extreme class imbalance  
✅ Threshold optimization based on business goals (F1 / Recall trade-offs)  
✅ Rich visual analytics dashboard  
✅ Robust model explainability (importance, permutation, PDP)  
✅ Fully deployed on Streamlit Cloud  

---

## 🧠 Problem Statement

Financial fraud detection is a highly imbalanced classification problem where: 

- Fraud cases are extremely rare
- False negatives are very costly
- High recall must be balanced with precision

This system detects fraudulent transactions with:

- Optimized decision threshold
- Interpretable outputs
- Production-grade stability

---

## 📂 Dataset

- **Source**: Credit Card Transactions Dataset
- **Records**: ~284,000 transactions
- **Fraud Rate**: ~0.17%
- **Features**:
  - `V1–V28`: PCA-transformed features
  - `Time`: Seconds since first transaction
  - `Amount`: Transaction amount
  - `Class`: Target variable (0 = Non-Fraud, 1 = Fraud)

---

## 🛠 Feature Engineering

The following additional features are engineered:

| Feature | Description |
|---------|-------------|
| `hour` | Hour of day extracted from Time |
| `day` | Day index extracted from Time |
| `amount_log` | Log-transformed transaction amount |
| `amount_log_scaled` | Standard-scaled log amount |

All features are aligned with the trained model's expected order. 

---

## 🤖 Model Details

- **Algorithm**: XGBoost Classifier
- **Objective**: Binary classification
- **Class Imbalance Handling**: `scale_pos_weight`
- **Evaluation Metrics**:
  - Precision
  - Recall
  - F1 Score
  - ROC-AUC

### 🔍 Baseline Performance

```
Accuracy : 0.9996
Precision: 0.9205
Recall   : 0.8265
F1 Score : 0.8710
AUC      : 0.9827
```

### 🚀 Improved Model (Class Imbalance Aware)

```
Accuracy : 0.9995
Precision: 0.8723
Recall   : 0.8367
F1 Score : 0.8542
AUC      :  0.9838
```

---

## 🎯 Threshold Optimization

Instead of using the default 0.5, thresholds were analyzed using:

- Precision–Recall curve
- F1 maximization
- Youden's J statistic

### ✅ Final Selected Threshold

```
Threshold = 0.8676844
```

This provides the best trade-off between precision and recall for real-world fraud detection.

---

## 📊 Streamlit Web Application

### 🔹 Pages Included

#### 1️⃣ Fraud Prediction

- Upload CSV of transactions
- **Outputs**:
  - Fraud probability
  - Binary fraud prediction
  - Downloadable results CSV

#### 2️⃣ Visual Analytics

- Fraud vs Non-Fraud distribution
- Hourly & daily fraud trends
- Heatmaps (Day × Hour)
- KDE plots
- Boxplots
- Correlation heatmaps

#### 3️⃣ Model Explainability

- XGBoost Gain-based Feature Importance
- Permutation Importance (model-agnostic)
- Manual Partial Dependence Plots (PDP)
- Stable and version-safe
- Shows marginal effect of features

#### 4️⃣ Model Performance

- Confusion Matrix
- ROC Curve & AUC
- Precision–Recall Curve
- Threshold tuning visualization
- Classification Report

#### 5️⃣ About Project

- System overview
- Key techniques used

---

## 🔍 Explainability Strategy

Due to cross-version compatibility issues with SHAP and serialized XGBoost models:

❌ SHAP disabled (intentionally)  
✅ Used robust alternatives: 

- Gain-based feature importance
- Permutation importance
- Manual PDP (production-safe)

This ensures no runtime crashes on deployment.

---

## ☁️ Deployment

- **Platform**: Streamlit Cloud
- **Main file**: `streamlit_app.py`
- **Artifacts loaded**: 
  - `fraud_xgb_model.pkl`
  - `fraud_scaler.pkl`

### Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## 📁 Project Structure

```
├── streamlit_app.py
├── fraud_xgb_model.pkl
├── fraud_scaler.pkl
├── requirements. txt
├── README.md
```

---

## ⚠️ Known Warnings (Handled Safely)

- XGBoost version mismatch warnings
- Scikit-learn unpickle version warnings

These do not affect predictions and are safely handled in code.

---

## 👨‍💻 Author

**Shailendra Bhushan Rai**  
B. Tech Computer Science & Engineering  
Data Scientist / ML Engineer
