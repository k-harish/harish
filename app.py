import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model
model = joblib.load('linear_regression_model.pkl')

# Title of the app
st.title("Urban Air Quality and Health Risk Prediction")

# User inputs for features
st.header("Input the feature values:")
temp = st.number_input("Temperature", value=25.0)
humidity = st.number_input("Humidity", value=50.0)
heat_index = st.number_input("Heat Index", value=30.0)
temp_range = st.number_input("Temperature Range", value=10.0)
severity_score = st.number_input("Severity Score", value=2.0)

# Prepare input data for prediction
input_data = pd.DataFrame({
    'temp': [temp],
    'humidity': [humidity],
    'Heat_Index': [heat_index],
    'Temp_Range': [temp_range],
    'Severity_Score': [severity_score]
})

# Make predictions
if st.button('Predict Health Risk'):
    prediction = model.predict(input_data)
    st.write(f"Predicted Health Risk Score: {prediction[0]:.2f}")

# Optionally, add some description or instructions
st.text("Adjust the feature values and click 'Predict Health Risk' to see the prediction.")
