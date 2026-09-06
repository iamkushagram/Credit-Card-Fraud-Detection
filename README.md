# 💳 Credit Card Fraud Detection

A Machine Learning-powered web application that detects fraudulent credit card transactions in real time. Built with **Streamlit** and deployed on **Streamlit Cloud**.

---

## 📌 Project Overview

Credit card fraud is a major financial threat. This app uses a trained machine learning model to predict whether a given transaction is **fraudulent or legitimate** based on transaction features.

---

## 📁 Project Structure

```
credit-card-fraud-detection/
├── pkl/
│   └── fraud_model.pkl          # Trained ML model
├── Model/
│   └── Credit_Card_Fraud_Detection.ipynb  # Training notebook
├── application.py               # Streamlit app
├── requirements.txt             # Python dependencies
├── runtime.txt                  # Python version (3.11)
└── README.md
```

---

## ✨ Features

- 🔍 **Single Transaction Prediction** — Enter transaction features manually and get an instant fraud/legitimate verdict
- 📊 **Batch Prediction** — Upload a CSV file to predict fraud across multiple transactions at once
- 📥 **Downloadable Results** — Download batch prediction results as a CSV
- ⚡ **Fast & Lightweight** — Cached model loading for quick inference

---

## 🤖 Model Details

| Property | Value |
|----------|-------|
| Algorithm | Random Forest / XGBoost |
| Training Library | scikit-learn |
| Dataset | Credit Card Fraud Detection (Kaggle) |
| Target Variable | `Class` (0 = Legitimate, 1 = Fraud) |

> **Note:** The dataset is highly imbalanced — fraudulent transactions account for only ~0.17% of all transactions.

---

## 🚀 Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/iamkushagram/Credit-Card-Fraud-Detection.git
cd credit-card-fraud-detection
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run application.py
```

---

## 📦 Requirements

```
streamlit>=1.35.0
pandas
numpy
scikit-learn
xgboost
```

---

## 📊 Dataset

The dataset used for training is the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle.

- **284,807** transactions
- **492** fraudulent cases
- Features are PCA-transformed (V1–V28) + `Time` and `Amount`

---

## 👨‍💻 Author

**Kushagra Mishra** — [GitHub](https://github.com/iamkushagram)