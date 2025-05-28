import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load model and feature columns
model = RandomForestClassifier()
model.load = model.fit  # To avoid pickle errors if re-using trained model
df = pd.read_csv("preprocessed_data.csv")
X = df.drop("Osteoporosis", axis=1)
y = df["Osteoporosis"]
model.fit(X, y)  # Refit the model for demo

# Title
st.title("🦴 Osteoporosis Prediction App")
st.write("Fill in the information below to predict the risk of Osteoporosis.")

# User inputs
age = st.slider("Age", 18, 100, 30)

gender = st.selectbox("Gender", ["Male", "Female"])
hormonal = st.selectbox("Hormonal Changes", ["Normal", "Postmenopausal"])
family = st.selectbox("Family History", ["Yes", "No"])
ethnicity = st.selectbox("Race/Ethnicity", ["Asian", "Caucasian", "African American"])
weight = st.selectbox("Body Weight", ["Underweight", "Normal"])
calcium = st.selectbox("Calcium Intake", ["Low", "Adequate"])
vit_d = st.selectbox("Vitamin D Intake", ["Insufficient", "Sufficient"])
activity = st.selectbox("Physical Activity", ["Sedentary", "Active"])
smoking = st.selectbox("Smoking", ["Yes", "No"])
alcohol = st.selectbox("Alcohol Consumption", ["Moderate", "None"])
condition = st.selectbox("Medical Conditions", ["Hyperthyroidism", "Rheumatoid Arthritis", "None"])
meds = st.selectbox("Medications", ["Corticosteroids", "None"])
fractures = st.selectbox("Prior Fractures", ["Yes", "No"])

# Encode inputs the same way as preprocessing
def encode_input():
    return pd.DataFrame([[
        age,
        1 if gender == "Male" else 0,
        0 if hormonal == "Normal" else 1,
        1 if family == "Yes" else 0,
        {"Asian":0, "Caucasian":1, "African American":2}[ethnicity],
        0 if weight == "Underweight" else 1,
        0 if calcium == "Low" else 1,
        0 if vit_d == "Insufficient" else 1,
        0 if activity == "Sedentary" else 1,
        1 if smoking == "Yes" else 0,
        0 if alcohol == "None" else 1,
        0 if condition == "Hyperthyroidism" else (1 if condition == "Rheumatoid Arthritis" else 2),
        0 if meds == "None" else 1,
        1 if fractures == "Yes" else 0
    ]], columns=X.columns)

# Predict button
if st.button("Predict"):
    input_df = encode_input()
    result = model.predict(input_df)
    st.subheader("🔮 Prediction Result")
    if result[0] == 1:
        st.error("High Risk of Osteoporosis ❗")
    else:
        st.success("Low Risk of Osteoporosis ✅")

# to run this code : 
#  python -m streamlit run app.py