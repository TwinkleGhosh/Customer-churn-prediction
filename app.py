# IMPORT LIBRARIES
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# LOAD MODEL FILES

model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# PAGE CONFIG

st.set_page_config(page_title="Churn Prediction", layout="wide")

# APP TITLE

st.title("📊 Customer Churn Prediction App")
st.write("Predict whether a customer will leave the service")

# Adding Tabs

tab1, tab2 = st.tabs(["📊 Prediction", "📈 Insights"])

# USER INPUT SECTION

st.sidebar.header("Enter Customer Details")

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges", value=50.0)
total_charges = st.sidebar.number_input("Total Charges", value=500.0)

# Dropdowns

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

# CREATE INPUT DATAFRAME

input_df = pickle.load(open("sample_input.pkl", "rb")).copy()

# Fill important features

input_df["tenure"] = tenure
input_df["MonthlyCharges"] = monthly_charges
input_df["TotalCharges"] = total_charges

input_df["Contract_Two year"] = 0
input_df["Contract_One year"] = 0

if contract == "One year":
    input_df["Contract_One year"] = 1
elif contract == "Two year":
    input_df["Contract_Two year"] = 1


# Gender encoding

if "gender_Male" in input_df.columns:
    input_df["gender_Male"] = 1 if gender == "Male" else 0

# Senior citizen

if "SeniorCitizen" in input_df.columns:
    input_df["SeniorCitizen"] = senior

# SCALE INPUT

input_scaled = input_df

# PREDICTION BUTTON

with tab1:

    st.subheader("Predict Customer Churn")

    if st.sidebar.button("Predict Churn"):

        prob = model.predict_proba(input_scaled)[0][1]

        st.subheader("📊 Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            if prob > 0.3:
                st.error("⚠️ High Risk of Churn")
            else:
                st.success("✅ Low Risk of Churn")

        with col2:
            st.metric("Churn Probability", f"{prob:.2f}")

with tab2:

    st.header("📊 Customer Insights")
    st.info("Customers with low tenure have higher churn probability.")
    st.image("tenure_distribution.png", caption="Tenure Distribution")
    st.image("tenure_vs_churn.png", caption="Tenure vs Churn")
    st.image("churn_rate.png", caption="Churn Rate by Tenure Group")

# FOOTER (OPTIONAL)

st.write("---")
st.write("Built using Machine Learning + Streamlit")
