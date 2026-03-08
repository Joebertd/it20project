import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# --- Configuration and File Paths ---
# Assuming these files are in the current working directory or a 'models'/'preprocessing' subdirectory
MODEL_PATH = 'models/logistic_model.pkl'
ENCODERS_PATH = 'preprocessing/encoders.pkl'
FEATURE_CONFIG_PATH = 'models/feature_config.json'

# --- Load Assets ---
@st.cache_resource
def load_model_assets():
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    with open(FEATURE_CONFIG_PATH, 'r') as f:
        feature_config = json.load(f)
    return model, encoders, feature_config

model, encoders, feature_config = load_model_assets()

# Extract components from loaded assets
scaler = encoders['scaler']
label_encoders = {k: v for k, v in encoders.items() if k != 'scaler'}

CATEGORICAL = feature_config['categorical']
NUMERICAL = feature_config['numerical']
ENGINEERED = feature_config['engineered']
FEATURE_COLS = feature_config['all_features']

# --- Helper Function for Feature Engineering ---
def preprocess_input(input_data):
    # Create a DataFrame from input_data for easier processing
    df_input = pd.DataFrame([input_data])

    feat = pd.DataFrame()

    # Re-apply feature engineering steps based on notebook logic
    feat['monthly_income']  = df_input['ApplicantIncome'] + df_input['CoapplicantIncome']
    feat['loan_amount']     = df_input['LoanAmount'] * 1000
    feat['loan_term']       = df_input['Loan_Amount_Term'].astype(int)

    # Credit_History to credit_score (simplified for demo)
    # In a real app, you might have a more robust way to derive this from user input.
    # For this demo, we'll map 1 to a high score, 0 to a low score, or ask directly.
    # Assuming Credit_History is 0 or 1 for simplicity here
    if df_input['Credit_History'].iloc[0] == 1:
        feat['credit_score'] = int(np.random.normal(700, 40))
    else:
        feat['credit_score'] = int(np.random.normal(480, 60))
    feat['credit_score'] = np.clip(feat['credit_score'], 300, 850)

    feat['existing_loans']  = df_input['Dependents'].astype(int)

    monthly_payment = feat['loan_amount'] / feat['loan_term'].clip(lower=1)
    feat['debt_to_income'] = (monthly_payment / feat['monthly_income'].clip(lower=1)).clip(0, 1)

    # Approximate age (simplified for demo)
    edu_age_map = {'Graduate': 28, 'Not Graduate': 24}
    base_age = df_input['Education'].map(edu_age_map)
    dep_add = df_input['Dependents'] * 2
    feat['age'] = (base_age + dep_add + np.random.randint(0, 12)).clip(22, 65).astype(int)

    feat['employment_status'] = np.where(df_input['Self_Employed'] == 'Yes', 'Self-Employed', 'Employed')
    feat['education'] = np.where(df_input['Education'] == 'Graduate', 'Bachelor', 'High School')
    feat['marital_status'] = np.where(df_input['Married'] == 'Yes', 'Married', 'Single')

    # Engineered features
    feat['income_loan_ratio']        = feat['monthly_income'] / feat['loan_amount'].clip(lower=1)
    feat['monthly_payment']          = feat['loan_amount']    / feat['loan_term'].clip(lower=1)
    feat['affordability_index']      = feat['monthly_income'] / feat['monthly_payment'].clip(lower=1)
    feat['credit_utilization_score'] = feat['credit_score']   / 850.0
    feat['loan_to_income_monthly']   = feat['monthly_payment'] / feat['monthly_income'].clip(lower=1)

    # Apply Label Encoding for categorical features
    for col in CATEGORICAL:
        original_col_name = col.replace('_enc', '') # Get original name if already _enc
        if original_col_name in label_encoders:
            le = label_encoders[original_col_name]
            feat[f'{col}_enc'] = le.transform(feat[col].astype(str))
        else:
             # Handle cases where the original col might not be in the direct map if different names
            feat[f'{col}_enc'] = label_encoders[col].transform(feat[col].astype(str)) if col in label_encoders else feat[col]

    # Select and order features as expected by the model
    X_processed = feat[FEATURE_COLS].values

    # Scale numerical features
    X_scaled = scaler.transform(X_processed)

    return X_scaled

# --- Streamlit UI ---
st.set_page_config(page_title="LoanIQ Loan Approval Predictor", layout="centered")
st.title(" LoanIQ Loan Approval Predictor")
st.markdown("Enter applicant details to predict loan approval status.")

# Input fields
st.subheader("Applicant Information")

gender = st.selectbox("Gender", ['Male', 'Female'])
married = st.selectbox("Married", ['Yes', 'No'])
dependents = st.number_input("Number of Dependents", min_value=0, max_value=3, value=0)
education = st.selectbox("Education", ['Graduate', 'Not Graduate'])
self_employed = st.selectbox("Self Employed", ['No', 'Yes'])
applicant_income = st.number_input("Applicant Income (USD)", min_value=0.0, value=5000.0, step=100.0)
coapplicant_income = st.number_input("Coapplicant Income (USD)", min_value=0.0, value=0.0, step=100.0)
loan_amount = st.number_input("Loan Amount (in thousands USD)", min_value=10.0, value=150.0, step=10.0)
loan_amount_term = st.selectbox("Loan Amount Term (months)", [12, 36, 60, 84, 120, 180, 240, 300, 360, 480], index=8)
credit_history = st.selectbox("Credit History (1=Good, 0=Bad)", [1, 0])
property_area = st.selectbox("Property Area", ['Urban', 'Semiurban', 'Rural'])


if st.button("Predict Loan Approval"):
    input_data = {
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': float(loan_amount_term),
        'Credit_History': float(credit_history),
        'Property_Area': property_area, # Not directly used by model, but for consistency
        'Loan_ID': 'N/A', # Placeholder
        'Loan_Status': 'N/A' # Placeholder
    }

    try:
        processed_input = preprocess_input(input_data)
        prediction = model.predict(processed_input)
        prediction_proba = model.predict_proba(processed_input)

        if prediction[0] == 1:
            st.success(f"**Prediction: Loan Approved!** (Probability: {prediction_proba[0][1]:.2f})")
        else:
            st.error(f"**Prediction: Loan Rejected.** (Probability: {prediction_proba[0][0]:.2f})")

        st.markdown("### Raw Input")
        st.json(input_data)
        st.markdown("### Processed Input Features (Scaled)")
        st.write(pd.DataFrame(processed_input, columns=FEATURE_COLS))

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Please ensure all input fields are correctly filled and files are accessible.")

st.markdown("--- Source: Logistic Regression Model --- ")
