import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("solar_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🌞 Solar Energy Prediction System")

# Input fields
temp = st.number_input("Temperature (°C)")
humidity = st.number_input("Humidity (%)")
cloud = st.number_input("Cloud Cover (%)")
irradiance = st.number_input("Irradiance (W/m²)")
wind = st.number_input("Wind Speed (m/s)")
angle = st.number_input("Incidence Angle (°)")
zenith = 50
azimuth = 180

if st.button("Predict Solar Energy Output"):
    features = np.array([[temp, humidity, cloud, irradiance, wind, angle, zenith, azimuth]])
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)[0]
    st.success(f"⚡ Predicted Solar Power Output: {prediction:.2f} kW")
