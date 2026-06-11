import streamlit as st
import pickle
import numpy as np

# Load saved model and scaler
rf_model = pickle.load(open('rf_model.pkl', 'rb'))
scaler   = pickle.load(open('scaler.pkl', 'rb'))

st.title("Loan Default Risk Predictor")
st.write("Fill in the applicant details below to predict loan risk.")

# Input Fields
person_age = st.number_input("Age", min_value=18, max_value=100, value=30)

person_income = st.number_input("Annual Income (RM)", min_value=0, value=50000)

person_home_ownership = st.selectbox(
    "Home Ownership",
    options=[0, 1, 2, 3],
    format_func=lambda x: {0: "Rent", 1: "Mortgage", 2: "Own", 3: "Other"}[x]
)

person_emp_length = st.number_input("Employment Length (years)", min_value=0, max_value=50, value=5)

loan_intent = st.selectbox(
    "Loan Intent",
    options=[0, 1, 2, 3, 4, 5],
    format_func=lambda x: {
        0: "Education", 1: "Medical", 2: "Venture",
        3: "Personal", 4: "Debt Consolidation", 5: "Home Improvement"
    }[x]
)

loan_grade = st.selectbox(
    "Loan Grade",
    options=[0, 1, 2, 3, 4, 5, 6],
    format_func=lambda x: {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G"}[x]
)

loan_amnt = st.number_input("Loan Amount (RM)", min_value=0, value=10000)

loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=30.0, value=10.0)

loan_percent_income = st.number_input("Loan as % of Income (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=0.2)

cb_person_default_on_file = st.selectbox(
    "Previous Default on Record?",
    options=[0, 1],
    format_func=lambda x: {0: "No", 1: "Yes"}[x]
)

cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, max_value=30, value=5)

#Predict Button
if st.button("Predict Risk"):

    # Arrange input in same column order as training data
    input_data = np.array([[
        person_age,
        person_income,
        person_home_ownership,
        person_emp_length,
        loan_intent,
        loan_grade,
        loan_amnt,
        loan_int_rate,
        loan_percent_income,
        cb_person_default_on_file,
        cb_person_cred_hist_length
    ]])

    # Scale using the same scaler from training
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction   = rf_model.predict(input_scaled)[0]
    probability  = rf_model.predict_proba(input_scaled)[0]

    default_prob  = probability[1] * 100  # probability of Default (1)
    paid_prob     = probability[0] * 100  # probability of Fully Paid (0)

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ HIGH RISK — Likely to Default")
    else:
        st.success(f"✅ LOW RISK — Likely to Fully Pay")

    st.write(f"Probability of Default:    **{default_prob:.1f}%**")
    st.write(f"Probability of Fully Paid: **{paid_prob:.1f}%**")