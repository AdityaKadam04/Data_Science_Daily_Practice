import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load("C:/Users/ushad/Desktop/Study/100_Days_of machine learning/1_Ml_project/Deploy/model.pkl")



# Title
st.title("Placement Prediction App")

# Input fields
iq = st.number_input("Enter IQ", min_value=0.0, max_value=200.0, step=0.1)
cgpa = st.number_input("Enter CGPA", min_value=0.0, max_value=10.0, step=0.1)

# Predict button
if st.button("Predict Placement"):
    input_data = np.array([[iq, cgpa]])
    prediction = model.predict(input_data)
    
    result = "Placed" if prediction[0] == 1 else "Not Placed"
    st.success(f"Prediction: {result}")
