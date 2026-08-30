import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------------------------------------------
# Page config
# -----------------------------------------------------------------
st.set_page_config(
    page_title="CreditWise | Loan Approval Predictor",
    page_icon="💳",
    layout="centered",
)

# -----------------------------------------------------------------
# Load custom CSS
# -----------------------------------------------------------------
def load_css(file_path: str):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# -----------------------------------------------------------------
# Load model artifacts (cached so they load only once)
# -----------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.joblib")
    scaler = joblib.load("scaler.joblib")
    ohe = joblib.load("ohe.joblib")
    le_education = joblib.load("le_education.joblib")
    le_target = joblib.load("le_target.joblib")
    feature_columns = joblib.load("feature_columns.joblib")
    ohe_cols = joblib.load("ohe_cols.joblib")
    return model, scaler, ohe, le_education, le_target, feature_columns, ohe_cols

model, scaler, ohe, le_education, le_target, feature_columns, ohe_cols = load_artifacts()

# -----------------------------------------------------------------
# Header
# -----------------------------------------------------------------
st.markdown(
    """
    <div class="cw-header">
        <h1>💳 CreditWise</h1>
        <p>An ML-powered loan approval predictor — fill in the applicant's details below.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------
# Input form
# -----------------------------------------------------------------
with st.form("loan_form"):

    st.markdown('<div class="cw-card"><h3>👤 Applicant Details</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married"])
    with col2:
        education_level = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
        dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="cw-card"><h3>💼 Employment & Income</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        employment_status = st.selectbox(
            "Employment Status", ["Salaried", "Self-employed", "Contract", "Unemployed"]
        )
        employer_category = st.selectbox(
            "Employer Category", ["Private", "Government", "MNC", "Business", "Unemployed"]
        )
    with col2:
        applicant_income = st.number_input("Applicant Monthly Income", min_value=0, value=15000, step=500)
        coapplicant_income = st.number_input("Coapplicant Monthly Income", min_value=0, value=0, step=500)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="cw-card"><h3>🏦 Financial Profile</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        credit_score = st.slider("Credit Score", min_value=300, max_value=900, value=650)
        dti_ratio = st.slider("Debt-to-Income (DTI) Ratio", min_value=0.0, max_value=1.0, value=0.30, step=0.01)
        existing_loans = st.number_input("Existing Loans", min_value=0, max_value=20, value=1)
    with col2:
        savings = st.number_input("Savings", min_value=0, value=10000, step=500)
        collateral_value = st.number_input("Collateral Value", min_value=0, value=20000, step=500)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="cw-card"><h3>📄 Loan Details</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        loan_amount = st.number_input("Loan Amount Requested", min_value=0, value=25000, step=500)
        loan_term = st.number_input("Loan Term (months)", min_value=1, max_value=480, value=60)
    with col2:
        loan_purpose = st.selectbox(
            "Loan Purpose", ["Personal", "Car", "Business", "Home", "Education"]
        )
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    st.markdown("</div>", unsafe_allow_html=True)

    submitted = st.form_submit_button("🔍 Predict Loan Approval")

# -----------------------------------------------------------------
# Prediction logic
# -----------------------------------------------------------------
if submitted:
    # Build a single-row dataframe matching the raw schema
    raw_input = pd.DataFrame([{
        "Applicant_Income": applicant_income,
        "Coapplicant_Income": coapplicant_income,
        "Employment_Status": employment_status,
        "Age": age,
        "Marital_Status": marital_status,
        "Dependents": dependents,
        "Credit_Score": credit_score,
        "Existing_Loans": existing_loans,
        "DTI_Ratio": dti_ratio,
        "Savings": savings,
        "Collateral_Value": collateral_value,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,
        "Loan_Purpose": loan_purpose,
        "Property_Area": property_area,
        "Education_Level": education_level,
        "Gender": gender,
        "Employer_Category": employer_category,
    }])

    # Label encode Education_Level
    raw_input["Education_Level"] = le_education.transform(raw_input["Education_Level"])

    # One-hot encode the same columns used at training time
    encoded = ohe.transform(raw_input[ohe_cols])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(ohe_cols), index=raw_input.index)
    processed = pd.concat([raw_input.drop(columns=ohe_cols), encoded_df], axis=1)

    # Feature engineering (must match training)
    processed["DTI_Ratio_sq"] = processed["DTI_Ratio"] ** 2
    processed["Credit_Score_sq"] = processed["Credit_Score"] ** 2
    processed = processed.drop(columns=["Credit_Score", "DTI_Ratio"])

    # Align columns exactly to the training feature order (fills any missing dummy cols with 0)
    processed = processed.reindex(columns=feature_columns, fill_value=0)

    # Scale and predict
    scaled_input = scaler.transform(processed)
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    label = le_target.inverse_transform([prediction])[0]

    if label == "Yes":
        st.markdown(
            f'<div class="cw-result-approved">✅ Loan Likely Approved<br>'
            f'<span style="font-size:0.95rem;font-weight:400;">Confidence: {probability*100:.1f}%</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="cw-result-rejected">❌ Loan Likely Rejected<br>'
            f'<span style="font-size:0.95rem;font-weight:400;">Confidence: {(1-probability)*100:.1f}%</span></div>',
            unsafe_allow_html=True,
        )

    st.caption(
        "⚠️ This prediction is generated by a machine learning model trained on a sample "
        "dataset for educational purposes. It is not financial advice and should not be "
        "used for real lending decisions."
    )

# -----------------------------------------------------------------
# Footer
# -----------------------------------------------------------------
st.markdown(
    '<div class="cw-footer">Built with ❤️ using Streamlit & Scikit-learn — CreditWise Project</div>',
    unsafe_allow_html=True,
)
