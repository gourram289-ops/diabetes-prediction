import streamlit as st 
import pandas as pd 
import joblib 
import requests

URL = "https://diabetes-prediction-ytm1.onrender.com"

st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title(' Diabetes Prediction App')
st.markdown('<div class="sub-header">Powered by a Scikit-Learn Preprocessing Pipeline</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ System Status")
    try:
        response = requests.get(f"{URL}/",timeout=5)
        if response.status_code == 200:
            st.success("✅ Model is loaded and the system is online.")
        else:
            st.warning("⚠️ Model is not loaded or the system is offline.")
    except requests.exceptions.RequestException:
        st.error("❌ Could not connect to the system. Please check your internet connection or try again later.")

st.markdown("---")

st.markdown('<div class="input-header">Input Parameters</div>', unsafe_allow_html=True)
st.write("Adjust the parameters below to dynamically estimate fair-market compensation based on current industry benchmarks.")
col1, col2 ,col3= st.columns(3)

with col1:
    st.header("Patient Information")

    gender = st.selectbox("gender",["male","female","other"])

    age = st.slider("age", 0, 100, 30)

with col2:
    st.header("Health Metrics")

    bmi = st.slider("bmi", 10.0, 100.0, 25.0)

    HbA1c_level = st.slider("HbA1c_level", 3.0, 20.0, 5.0)

    blood_glucose_level = st.number_input("blood_glucose_level", 50, 300, 100)
with col3:
    st.header("Medical History")

    hypertension = st.selectbox("hypertension",["0","1"])

    heart_disease = st.selectbox("heart_disease",["0","1"])

    smoking_history = st.selectbox("smoking_history",["never","ever","current","former","not current","unknown"])


st.markdown("---")
if st.button("⚡ Calculate Predicted Compensation", use_container_width=True, type="primary"):
    playload = {
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "smoking_history": smoking_history,
        "bmi": bmi,
        "HbA1c_level": HbA1c_level,
        "blood_glucose_level": blood_glucose_level
    }

    with st.spinner("Calculating..."):
        try:
            response = requests.post(f"{URL}/predict",json = playload,timeout=2)
            if response.status_code == 200:
                data = response.json()
                result  = data.get("prediction")
                if result == 1:
                    st.warning(" The patient is predicted to have diabetes.")
                else:
                    st.success("✅ The patient is not predicted to have diabetes.")
            else:
                st.error("❌ Failed to get a valid response from the server. Please try again later.")  
        except requests.exceptions.RequestException:
            st.error("❌ Could not connect to the system. Please check your internet connection or try again later.")


