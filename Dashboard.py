import streamlit as st
import pandas as pd

st.set_page_config(page_title="Dataset Preview", layout="centered")
st.title("📄 Local Dataset Preview")

# ✅ Load from local folder (simple and clean)
try:
    df = pd.read_csv("datasets/sherbrooke_fixed_sensor_readings.csv", nrows=5)
    data2 = pd.read_csv("datasets/sherbrooke_sensor_readings_with_anomalies.csv", nrows=5)

    st.subheader("✅ Fixed Dataset Preview")
    st.dataframe(df)

    st.subheader("✅ Anomalies Dataset Preview")
    st.dataframe(data2)

except Exception as e:
    st.error(f"❌ Failed to load dataset: {e}")
