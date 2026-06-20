import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("random_forest_model.pkl")

st.title("Boston Housing Price Prediction")
st.write("Enter the housing features to predict the median value (MEDV) in $ 000's")

# Input fields for all predictors
crim = st.number_input("Per capita crime rate (crim)", min_value=0.0,  format="%.5f",
    step=0.00001)
rm = st.number_input("Average number of rooms (rm)", min_value=0.0, format="%.3f",
    step=0.001)
dis = st.number_input("Distance to employment centers (dis)", min_value=0.0, format="%.4f",
    step=0.0001)
tax = st.number_input("Property tax rate (tax)", min_value=0.0)
lstat = st.number_input("Lower status population % (lstat)", min_value=0.0)

if st.button("Predict"):
    features = np.array([[crim, rm, dis,tax,lstat]])
    prediction = model.predict(features)
    st.success(f"Predicted Median Value (MEDV): {prediction[0]:.2f}")
