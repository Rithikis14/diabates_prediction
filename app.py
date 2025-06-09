import streamlit as st
import pandas as pd
import pickle

# Load the trained model
model = pickle.load(open('diabetes_model.pkl', 'rb'))

# Title
st.title("🩺 Diabetes Prediction Web App")

# Input Fields
st.header("Enter Patient Details:")

pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=1)

# Predict button
if st.button("Predict"):
    # Input as DataFrame (with column names to avoid warnings)
    input_df = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness,
                              insulin, bmi, dpf, age]],
                            columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                                     'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

    # Make prediction
    result = model.predict(input_df)

    # Display result
    if result[0] == 1:
        st.error("⚠️ The model predicts: **Diabetes Positive**.")
    else:
        st.success("✅ The model predicts: **Diabetes Negative**.")
