import streamlit as st
import pandas as pd

st.set_page_config(page_title="Test Dropbox Read", layout="centered")

st.title("📦 Preview Dataset from Dropbox")

# ✅ Raw direct link (update if needed)
dropbox_url = "https://dl.dropboxusercontent.com/s/zaf92qddhz0wkiqjxwv0m/sherbrooke_fixed_sensor_readings.csv"

try:
    df = pd.read_csv(dropbox_url, on_bad_lines='skip')
    st.success("Dataset loaded successfully ✅")
    st.write("### Preview of the dataset:")
    st.dataframe(df.head())  # only show first few rows
except Exception as e:
    st.error(f"❌ Failed to load dataset: {e}")
