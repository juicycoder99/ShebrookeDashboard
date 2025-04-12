import streamlit as st
import pandas as pd
import os

# Set page config
st.set_page_config(page_title="Preview Sherbrooke Dataset", layout="wide")

st.markdown("## 📄 Preview Datasets from Kaggle")

# ✅ Add Kaggle credentials
os.environ['KAGGLE_USERNAME'] = 'jibrilhussaini'
os.environ['KAGGLE_KEY'] = 'your_kaggle_key_here'  # Replace this with your actual key from kaggle.json

@st.cache_data
def download_and_load():
    import zipfile
    import kaggle

    # ✅ Download dataset from Kaggle
    kaggle.api.dataset_download_files('jibrilhussaini/synthetic-sherbrooke-sensor-readings', path='data', unzip=True)

    # ✅ Load the CSVs
    df = pd.read_csv('data/sherbrooke_fixed_sensor_readings.csv')
    data2 = pd.read_csv('data/sherbrooke_sensor_readings_with_anomalies.csv')

    return df.head(), data2.head()

# ✅ Run the loader
try:
    fixed_df, anomalies_df = download_and_load()

    st.success("✅ Fixed Dataset Preview")
    st.dataframe(fixed_df)

    st.success("✅ Anomalies Dataset Preview")
    st.dataframe(anomalies_df)

except Exception as e:
    st.error(f"❌ Failed to load dataset: {e}")
