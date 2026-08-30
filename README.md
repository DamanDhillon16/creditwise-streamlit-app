# 💳 CreditWise — Loan Approval Predictor (Streamlit App)

A live, interactive web app that predicts loan approval using a Logistic Regression model trained on applicant financial data. This is the deployable companion to the [CreditWise ML notebook project](#) — same model, same preprocessing pipeline, wrapped in a simple UI.

🔗 **Live demo:** `<add your Streamlit Cloud link here after deployment>`

## 🖥️ What it does

Fill in an applicant's details — income, credit score, employment, loan amount, etc. — and the app predicts whether the loan is **likely to be approved or rejected**, along with a confidence score.

## 📁 Files in this repo

| File | Purpose |
|---|---|
| `app.py` | The Streamlit app (UI + prediction logic) |
| `style.css` | Custom styling for the app |
| `train_model.py` | Script that reproduces the model from raw data |
| `loan_approval_data.csv` | Training dataset |
| `model.joblib` | Trained Logistic Regression model |
| `scaler.joblib` | StandardScaler fitted on training data |
| `ohe.joblib` | OneHotEncoder fitted on categorical columns |
| `le_education.joblib`, `le_target.joblib` | Label encoders |
| `feature_columns.joblib`, `ohe_cols.joblib` | Column schema used to align inputs at inference time |
| `requirements.txt` | Python dependencies |

## 🚀 Run locally

```bash
git clone https://github.com/<your-username>/creditwise-streamlit-app.git
cd creditwise-streamlit-app
pip install -r requirements.txt
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## 🔁 Retrain the model (optional)

If you want to regenerate the model artifacts from scratch:

```bash
python train_model.py
```

This reruns the full preprocessing + training pipeline and overwrites the `.joblib` files.

## 🛠️ Tech Stack

Streamlit · Scikit-learn · Pandas · NumPy

## ⚠️ Disclaimer

This app is for educational purposes only. Predictions are based on a small sample dataset and should not be used for real lending or financial decisions.
