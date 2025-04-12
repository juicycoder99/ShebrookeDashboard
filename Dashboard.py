import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Set Streamlit page config
st.set_page_config(page_title="Temperature, Humidity, Moisture & Gas Dashboard", layout="wide")

# Dashboard Header
st.markdown("""
    <h1 style='text-align: center; color: white; background-color: #176B87; padding: 15px; border-radius: 10px;'>
        Temperature, Humidity, Moisture & Gas Dashboard
    </h1>
""", unsafe_allow_html=True)

# ✅ Cached data loader and preprocessor
@st.cache_data
def load_and_preprocess():
    try:
        # Load local CSVs
        df = pd.read_csv("datasets/sherbrooke_fixed_sensor_readings.csv", on_bad_lines='skip')
        data2 = pd.read_csv("datasets/sherbrooke_sensor_readings_with_anomalies.csv", on_bad_lines='skip')
    except Exception as e:
        st.error(f"❌ Error loading datasets: {e}")
        return pd.DataFrame(), pd.DataFrame()

    # Function to clean and process each dataset
    def process_data(data):
        # Combine Date and Time
        if 'Date' in data.columns and 'Time' in data.columns:
            data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], errors='coerce')
            data.drop(columns=['Date', 'Time'], inplace=True)

        # Encode Gas_Level as numeric
        if 'Gas_Level' in data.columns:
            data['Gas_Level'] = data['Gas_Level'].astype('category').cat.codes

        # Encode Location if exists
        if 'Location' in data.columns:
            data['Location_Code'] = data['Location'].astype('category').cat.codes

        # Drop missing and sort by datetime
        data.dropna(inplace=True)
        if 'Datetime' in data.columns:
            data.sort_values(by='Datetime', inplace=True)

        return data.reset_index(drop=True)

    # Apply preprocessing
    df = process_data(df)
    data2 = process_data(data2)

    return df, data2


# ✅ Load the preprocessed data
df, data2 = load_and_preprocess()

# ========== 📊 SIDEBAR SECTION ==========
st.sidebar.header("📂 Dataset Information")

# ✅ Display shape of loaded data
st.sidebar.success(f"✅ Normal File Shape: {df.shape}")
st.sidebar.success(f"✅ Anomaly File Shape: {data2.shape}")

# ✅ Real-time Clock in Sidebar
from datetime import datetime
st.sidebar.markdown(f"**🕒 Current Time:** {datetime.now().strftime('%I:%M:%S %p')}")

# ✅ Dataset toggle (normal vs anomalies)
dataset_choice = st.sidebar.radio("📁 Select Dataset:", ["Normal Readings", "Anomalies"])
data = df if dataset_choice == "Normal Readings" else data2

