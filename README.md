# 💳 AI-Based Financial Fraud Detection System

A complete end-to-end machine learning system for detecting fraudulent financial transactions.  
This project uses XGBoost, engineered features, and a Streamlit web app to score transactions in real-time.

---

## 🚀 Features

- End-to-end ML pipeline: preprocessing, feature engineering, training, threshold tuning.
- XGBoost-based fraud classifier optimized for highly imbalanced datasets.
- Engineered features:
  - Log-transformed amount
  - `hour` and `day` extracted from timestamps
  - Scaled `amount_log`
- Interactive Streamlit dashboard:
  - CSV upload
  - Fraud probability scoring
  - Ranking of suspicious transactions

---

## 🛠️ Tech Stack

- Python  
- XGBoost  
- Scikit-Learn  
- Pandas / NumPy  
- Streamlit  
- Joblib  

---

## 📁 Project Structure

AI-Based-Financial-Fraud-Detection-System/
 - fraud_xgb_model.pkl       # Trained XGBoost model
 - fraud_scaler.pkl          # StandardScaler for amount_log
 - fraud_threshold.txt       # Optimal probability threshold
 - streamlit_app.py          # Streamlit dashboard
 - test_sample.csv           # Example input file
 - README.md                 # Project documentation


---

## ▶️ Running the App

### 1️⃣ Install dependencies


pip install -r requirements.txt


### 2️⃣ Run the Streamlit application


streamlit run streamlit_app.py


---

## 📈 Future Enhancements

- SHAP explainability for model insights  
- Fraud pattern visualization  
- API deployment  
- Cloud-based real-time scoring system  

---

## 👨‍💻 Author

**Shailendra Rai**  
Aspiring Data Scientist | Java + ML Enthusiast  
GitHub: [Shailendra17103](https://github.com/Shailendra17103)
