import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("solar_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# App Title
st.set_page_config(page_title="Solar Energy Prediction", page_icon="🌞", layout="centered")
st.title("🌞 Solar Energy Prediction System")
st.markdown("Enter the environmental parameters below to predict solar power output (kW).")

# --- Input fields (two per row) ---
col1, col2 = st.columns(2)
with col1:
    temp = st.number_input("🌡️ Temperature (°C)", step=0.1)
with col2:
    humidity = st.number_input("💧 Humidity (%)", step=0.1)

col3, col4 = st.columns(2)
with col3:
    cloud = st.number_input("☁️ Cloud Cover (%)", step=0.1)
with col4:
    irradiance = st.number_input("🔆 Irradiance (W/m²)", step=0.1)

col5, col6 = st.columns(2)
with col5:
    wind = st.number_input("🌬️ Wind Speed (m/s)", step=0.1)
with col6:
    angle = st.number_input("📐 Incidence Angle (°)", step=0.1)

# Default values
zenith = 50
azimuth = 180

# --- Prediction button ---
st.markdown("---")
if st.button("🔮 Predict Solar Energy Output", use_container_width=True):
    features = np.array([[temp, humidity, cloud, irradiance, wind, angle, zenith, azimuth]])
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)[0]
    st.success(f"⚡ **Predicted Solar Power Output:** {prediction:.2f} kW")

# Footer
st.markdown("<br><center>Developed with ❤️ using Streamlit & Machine Learning</center>", unsafe_allow_html=True)
