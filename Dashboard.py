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
def load_and_preprocess():
    import kaggle

    # Set Kaggle credentials
    os.environ['KAGGLE_USERNAME'] = 'jibrilhussaini'
    os.environ['KAGGLE_KEY'] = 'your_kaggle_key_here'  # 🔐 Don't forget to replace!

    # Download and extract
    kaggle.api.dataset_download_files(
        'jibrilhussaini/synthetic-sherbrooke-sensor-readings',
        path='data',
        unzip=True
    )

    # Load both datasets
    df = pd.read_csv('data/sherbrooke_fixed_sensor_readings.csv', on_bad_lines='skip')
    data2 = pd.read_csv('data/sherbrooke_sensor_readings_with_anomalies.csv', on_bad_lines='skip')

    # Combine date + time into datetime
    for d in [df, data2]:
        if 'Date' in d.columns and 'Time' in d.columns:
            d['Datetime'] = pd.to_datetime(d['Date'] + ' ' + d['Time'], errors='coerce')
            d.drop(columns=['Date', 'Time'], inplace=True)
        elif 'Datetime' not in d.columns:
            st.warning("⚠️ Datetime column missing!")

        # Set Datetime as index
        d.set_index('Datetime', inplace=True)

        # Encode categorical column if it exists
        if 'Gas_Level' in d.columns:
            d['Gas_Level'] = d['Gas_Level'].astype('category').cat.codes

try:
    # Drop nulls
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



# ---------------------------- CSV Export Section ----------------------------

# 🔁 Convert selected data (normal or anomalies) to CSV
csv = data.to_csv(index=False).encode('utf-8')

# 📥 Download button inside expandable section
with st.sidebar.expander("📥 Download Reports", expanded=False):
    st.markdown("Export the currently selected dataset as a downloadable CSV file.")
    st.download_button(
        label="⬇️ Download CSV Report",
        data=csv,
        file_name='sensor_data_report.csv',
        mime='text/csv',
        use_container_width=True
    )


# ----------------------- Real-Time Sensor Overview Section -----------------------

st.markdown("## 🌡️ Real-Time Sensor Overview")

# ✅ Initialize random sensor sample row (only if not already done)
if 'random_row' not in st.session_state or data.empty:
    if not data.empty:
        st.session_state.random_row = data.sample(1).iloc[0]
        st.session_state.last_update = datetime.now()
    else:
        st.warning("⚠️ Dataset is empty. Cannot display sensor snapshot.")
        st.stop()

# 🔁 Manual refresh button for simulating real-time sensor check
if st.button("🔁 Refresh Sensor Data"):
    if not data.empty:
        st.session_state.random_row = data.sample(1).iloc[0]
        st.session_state.last_update = datetime.now()
    else:
        st.warning("⚠️ No data available to refresh.")
        st.stop()

# 🕒 Show last updated timestamp
if 'last_update' in st.session_state:
    st.caption(f"🕒 Last updated: {st.session_state.last_update.strftime('%Y-%m-%d %I:%M:%S %p')}")



# ----------------------- Realtime Sensor Metrics Display -----------------------

# 📊 Display key metrics for selected row
random_row = st.session_state.random_row
cols = st.columns(4)

cols[0].metric(label=f" Temperature ({random_row['Location']})", value=f"{round(random_row['Temperature'], 2)} °C", delta="Last update")
cols[1].metric(label=f" Humidity ({random_row['Location']})", value=f"{round(random_row['Humidity'], 2)} %", delta="Last update")
cols[2].metric(label=f" Moisture ({random_row['Location']})", value=f"{round(random_row['Moisture'], 2)}", delta="Last update")
cols[3].metric(label=f" Gas ({random_row['Location']})", value=f"{round(random_row['Gas'], 2)}", delta="Last update")


# ----------------------- Trend Visualizer Section -----------------------

# 📈 Dropdown to choose gas visualization mode
plot_option = st.selectbox("📈 Select Gas Level Trend View:", [
    "Select an option", 
    "Seasonal Average", 
    "Monthly Trend", 
    "Day vs Night Gas Levels", 
    "Sensor-wise Comparison"
])

# ➤ 1. Seasonal Average Gas Levels
if plot_option == "Seasonal Average":
    data['Season'] = data.index.month.map({
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Fall", 10: "Fall", 11: "Fall"
    })

    seasonal_trends = data.groupby('Season')['Gas'].mean().reindex(["Spring", "Summer", "Fall", "Winter"])

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=seasonal_trends.index, y=seasonal_trends.values, palette="coolwarm", ax=ax)
    ax.set_title("🌤️ Average Gas Levels Across Seasons")
    ax.set_xlabel("Season")
    ax.set_ylabel("Average Gas Level")
    st.pyplot(fig)


# ➤ 2. Monthly Gas Level Trend
elif plot_option == "Monthly Trend":
    monthly_trends = data.groupby(data.index.month)['Gas'].mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x=monthly_trends.index, y=monthly_trends.values, marker="o", ax=ax)
    ax.set_title("📆 Monthly Gas Level Trends")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Gas Level")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.grid(True)
    st.pyplot(fig)


# ➤ 3. Day vs Night Comparison
elif plot_option == "Day vs Night Gas Levels":
    data['Hour'] = data.index.hour
    data['TimeOfDay'] = data['Hour'].apply(lambda x: 'Day (6AM–6PM)' if 6 <= x < 18 else 'Night (6PM–6AM)')

    day_night_avg = data.groupby('TimeOfDay')['Gas'].mean().reindex(['Day (6AM–6PM)', 'Night (6PM–6AM)'])

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=day_night_avg.index, y=day_night_avg.values, palette="Set2", ax=ax)
    ax.set_title("🌗 Gas Levels: Day vs Night")
    ax.set_ylabel("Average Gas Level")
    st.pyplot(fig)


# ➤ 4. Sensor-wise Gas Level Comparison
elif plot_option == "Sensor-wise Comparison":
    top_n = 20
    sensor_avg = data.groupby('Location')['Gas'].mean().sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=sensor_avg.values, y=sensor_avg.index, palette="viridis", ax=ax)

    for i, (value, label) in enumerate(zip(sensor_avg.values, sensor_avg.index)):
        ax.text(value + 0.3, i, f"{value:.2f}", va='center', fontsize=9)

    ax.set_title(f"📍 Top {top_n} Sensor Locations by Gas Level")
    ax.set_xlabel("Average Gas Level")
    ax.set_ylabel("Sensor Location")
    plt.tight_layout()
    st.pyplot(fig)


# ➤ Default Info Message
elif plot_option == "Select an option":
    st.info("ℹ️ Please select a gas trend view to begin visualization.")

