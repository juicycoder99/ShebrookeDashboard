import streamlit as st
import pandas as pd
import gdown

st.set_page_config(page_title="Dataset Preview", layout="centered")

st.title("📄 Preview Datasets from Google Drive")

# Google Drive File IDs
file_id_1 = "1dL3siMY6KaX1z0f6C5GVgTlJ06b7_Wru"  # fixed dataset
file_id_2 = "1CHO_ToDIw7EET0TfAb1xOV4VynIYPrh8"  # anomalies dataset

# Construct download URLs
url1 = f"https://drive.google.com/uc?id={file_id_1}"
url2 = f"https://drive.google.com/uc?id={file_id_2}"

# Download and load just the first few rows
try:
    gdown.download(url1, "fixed.csv", quiet=True)
    gdown.download(url2, "anomalies.csv", quiet=True)

    df = pd.read_csv("fixed.csv", nrows=5)
    data2 = pd.read_csv("anomalies.csv", nrows=5)

    st.subheader("✅ Fixed Dataset Preview")
    st.dataframe(df)

    st.subheader("✅ Anomalies Dataset Preview")
    st.dataframe(data2)

except Exception as e:
    st.error(f"Failed to load datasets: {e}")
