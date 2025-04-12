import streamlit as st
import pandas as pd
import gdown

st.set_page_config(page_title="Preview Datasets from Google Drive", layout="centered")
st.title("📄 Preview Datasets from Google Drive")

# Google Drive file IDs
file_id_1 = "1dL3siMY6KaX1z0f6C5GVgTlJ06b7_Wru"
file_id_2 = "1CHO_ToDIw7EET0TfAb1xOV4VynIYPrh8"

# Proper Google Drive URLs
url1 = f"https://drive.google.com/uc?id={file_id_1}"
url2 = f"https://drive.google.com/uc?id={file_id_2}"

try:
    # Force gdown to handle redirect/confirmation with fuzzy=True
    gdown.download(url1, "fixed.csv", quiet=True, fuzzy=True)
    gdown.download(url2, "anomalies.csv", quiet=True, fuzzy=True)

    # Only show first 5 rows
    df = pd.read_csv("fixed.csv", nrows=5)
    data2 = pd.read_csv("anomalies.csv", nrows=5)

    st.subheader("✅ Fixed Dataset Preview")
    st.dataframe(df)

    st.subheader("✅ Anomalies Dataset Preview")
    st.dataframe(data2)

except Exception as e:
    st.error(f"❌ Failed to load datasets: {e}")
