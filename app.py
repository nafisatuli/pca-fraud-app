import streamlit as st
import joblib
import numpy as np
import pandas as pd

# -----------------------------
# Load components
# -----------------------------
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")
threshold = joblib.load("threshold.pkl")

st.title("💳 Credit Card Fraud Detection Demo")

st.write("This system evaluates whether a transaction is potentially fraudulent.")

# -----------------------------
# Select Mode
# -----------------------------
mode = st.radio("Select Input Mode:", ["Random Transaction", "Manual Entry"])

# -----------------------------
# RANDOM TRANSACTION MODE
# -----------------------------
if mode == "Random Transaction":

    df_sample = pd.read_csv("sample_transactions.csv")

    if st.button("Generate Random Transaction"):

        sample = df_sample.sample(1)

        input_array = sample.values

        probability = model.predict_proba(input_array)[0][1]

        st.subheader("Prediction Result")
        st.write(f"Fraud Probability: {probability:.4f}")

        if probability > threshold:
            st.error("⚠ High Fraud Risk Detected")
        else:
            st.success("✅ Low Fraud Risk")

        st.write("Transaction Data:")
        st.dataframe(sample)

# -----------------------------
# MANUAL ENTRY MODE
# -----------------------------
elif mode == "Manual Entry":

    st.subheader("Enter Transaction Details")

    # Basic visible inputs
    time = st.number_input("Transaction Time (seconds)", value=0.0)
    amount = st.number_input("Transaction Amount", value=0.0)

    # Advanced PCA inputs hidden
    with st.expander("Advanced PCA Features (Optional)"):
        st.write("These are internal model features.")
        pca_features = []
        for i in range(1, 29):
            value = st.number_input(f"V{i}", value=0.0)
            pca_features.append(value)

    # If user does not expand, default values will be zeros
    if 'pca_features' not in locals():
        pca_features = [0.0] * 28

    if st.button("Predict Fraud Risk"):

        scaled_amount = scaler.transform([[amount]])[0][0]

        input_data = [time] + pca_features + [scaled_amount]
        input_array = np.array(input_data).reshape(1, -1)

        probability = model.predict_proba(input_array)[0][1]

        st.subheader("Prediction Result")
        st.write(f"Fraud Probability: {probability:.4f}")

        if probability > threshold:
            st.error("⚠ High Fraud Risk Detected")
        else:
            st.success("✅ Low Fraud Risk")