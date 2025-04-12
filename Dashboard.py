import streamlit as st
import pandas as pd
import os
from datetime import datetime

# ---------------------------------------------
# 🔧 Streamlit Config & UI
# ---------------------------------------------
st.set_page_config(page_title="Temperature, Humidity, Moisture & Gas Dashboard", layout="wide")

st.markdown("""
    <h1 style='text-align: center; color: white; background-color: #176B87; padding: 15px; border-radius: 10px;'>
        Temperature, Humidity, Moisture & Gas Dashboard
    </h1>
""", unsafe_allow_html=True)

# ---------------------------------------------
# 🔐 Set Kaggle credentials securely
# ---------------------------------------------
os.environ['KAGGLE_USERNAME'] = 'jibrilhussaini'
os.environ['KAGGLE_KEY'] = 'your_kaggle_key_here'  # Replace with your actual Kaggle key

# ---------------------------------------------
# 📦 Download and Preprocess Data
# ---------------------------------------------
@st.cache_data
def download_and_preprocess():
    import kaggle

    try:
        kaggle.api.dataset_download_files(
            'jibrilhussaini/synthetic-sherbrooke-sensor-readings',
            path='data', unzip=True
        )

        df = pd.read_csv("data/sherbrooke_fixed_sensor_readings.csv", on_bad_lines='skip')
        data2 = pd.read_csv("data/sherbrooke_sensor_readings_with_anomalies.csv", on_bad_lines='skip')

        # Combine Date + Time
        if 'Date' in df.columns and 'Time' in df.columns:
            df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
            df.drop(columns=['Date', 'Time'], inplace=True)

        if 'Date' in data2.columns and 'Time' in data2.columns:
            data2['Datetime'] = pd.to_datetime(data2['Date'] + ' ' + data2['Time'], errors='coerce')
            data2.drop(columns=['Date', 'Time'], inplace=True)

        # Encode Gas_Level
        for d in [df, data2]:
            if 'Gas_Level' in d.columns:
                d['Gas_Level'] = d['Gas_Level'].astype('category').cat.codes
            d.dropna(inplace=True)

        return df, data2

    except Exception as e:
        st.error(f"❌ Error loading datasets: {e}")
        return pd.DataFrame(), pd.DataFrame()

# ---------------------------------------------
# 🚀 Load the datasets
# ---------------------------------------------
df, data2 = download_and_preprocess()



# ---------------------------------------------
# 📊 Sidebar Info & Dataset Selector
# ---------------------------------------------

# 🕒 Show current time
st.sidebar.markdown(f" **Current Time:** {datetime.now().strftime('%I:%M:%S %p')}")

# Add space between clock and dataset selector
st.sidebar.markdown(" ")

# 📁 Dataset selector
st.sidebar.markdown("📂 **Select Dataset:**")
dataset_choice = st.sidebar.radio(label="", options=["Normal Readings", "Readings with Anomalies"], label_visibility="collapsed")


# Dynamically assign selected dataset to `data`
data = df if dataset_choice == "Normal Readings" else data2



# ---------------------------------------
# 🧾 Sidebar Expandable Dashboard Info
# ---------------------------------------
with st.sidebar.expander("📄 Dashboard Info", expanded=False):
    st.markdown("""
    ### ℹ️ About This Dashboard

    This interactive dashboard is designed to:
    - Monitor real-time environmental sensor data  
    - Analyze gas levels across different timeframes  
    - Visualize temperature, humidity, moisture, and gas trends  
    - Explore correlations between gas and other variables  
    - Compare gas levels across sensor locations  
    - Detect and observe anomalies  
    - Drill into selected timeframes for insights

    ---

    ### 📊 Dataset Info  
    This is **synthetic** data — generated to simulate realistic environmental behavior.  
    Intended solely for **research and analysis** purposes.

    ---

    ### 🛠️ Technologies Used
    - Python, Pandas, NumPy  
    - Matplotlib, Seaborn  
    - Streamlit (UI framework)

    ---

    ### 👥 Credits
    Developed by: *Jibril Hussaini*  
    Supervised by: *Rachid Hedjam*  
    Institution: *Bishop’s University*  
    Year: 2025

    ---

    ### 📤 Future Add-ons
    - PDF/Excel reports  
    - Upload sensor logs  
    - AI-powered prediction  
    - Real-time map-based view
    """)



# ---------------------------------------
# 📈 Optional Summary Statistics Section
# ---------------------------------------
show_summary = st.sidebar.checkbox("📊 Show Statistical Summary")

if show_summary:
    st.markdown("## 📌 Statistical Summary")

    # Compute and format descriptive statistics
    summary = data[["Temperature", "Humidity", "Moisture", "Gas"]].describe().T.round(2)

    # Rename columns for clarity
    summary.rename(columns={
        "count": "Count",
        "mean": "Mean",
        "std": "Std Dev",
        "min": "Min",
        "25%": "25%",
        "50%": "Median",
        "75%": "75%",
        "max": "Max"
    }, inplace=True)

    # Display styled summary
    st.dataframe(summary.style.format("{:.2f}"))

