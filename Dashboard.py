import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from datetime import datetime, timedelta
import gdown

# Set Streamlit page configuration 
st.set_page_config(page_title="Temperature, Humidity, Moisture & Gas Dashboard", layout="wide")

# Add a header with title
st.markdown("""
    <h1 style='text-align: center; color: white; background-color: #176B87; padding: 15px; border-radius: 10px;'>
        Temperature, Humidity, Moisture & Gas Dashboard
    </h1>
""", unsafe_allow_html=True)

# ✅ Cached loader + preprocessor
@st.cache_data
def load_and_preprocess():
    import os
    import requests
    import pandas as pd

    def download_from_drive(file_id, destination):
        # Handles download of large files with confirmation tokens
        URL = "https://docs.google.com/uc?export=download"

        session = requests.Session()
        response = session.get(URL, params={'id': file_id}, stream=True)
        token = get_confirm_token(response)

        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(URL, params=params, stream=True)

        save_response_content(response, destination)

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive chunks
                    f.write(chunk)

    # ✅ Define file IDs and names
    files = {
        "fixed.csv": "1dL3siMY6KaX1z0f6C5GVgTlJ06b7_Wru",
        "anomalies.csv": "1CHO_ToDIw7EET0TfAb1xOV4VynIYPrh8"
    }

    # ✅ Download files
    for name, fid in files.items():
        if not os.path.exists(name):  # skip re-download if already exists
            download_from_drive(fid, name)

    # ✅ Load files
    try:
        df = pd.read_csv("fixed.csv", on_bad_lines='skip')
        data2 = pd.read_csv("anomalies.csv", on_bad_lines='skip')
    except Exception as e:
        st.error(f"❌ Error reading CSVs: {e}")
        return pd.DataFrame(), pd.DataFrame()

    # ✅ Combine datetime
    if 'Date' in df.columns and 'Time' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
        df.drop(columns=['Date', 'Time'], inplace=True)

    # ✅ Encode and clean
    for d in [df, data2]:
        if 'Gas_Level' in d.columns:
            d['Gas_Level'] = d['Gas_Level'].astype('category').cat.codes
        d.dropna(inplace=True)

    return df, data2






# ✅ Load the data once
df, data2 = load_and_preprocess()


st.sidebar.success(f"Normal File Shape: {df.shape}")
st.sidebar.success(f"Anomaly File Shape: {data2.shape}")


# 🕒 Sidebar clock
st.sidebar.markdown(f" **Current Time:** {datetime.now().strftime('%I:%M:%S %p')}")

# 📦 Dataset selector
dataset_choice = st.sidebar.radio("📁 Select Dataset:", ["Normal Readings", "Anomalies"])
data = df if dataset_choice == "Normal Readings" else data2


st.sidebar.markdown("## 🧪 Debug Info")

# Check file sizes
import os
st.sidebar.write("📦 Normal file size (MB):", round(os.path.getsize("sherbrooke_fixed_sensor_readings.csv") / 1e6, 2))
st.sidebar.write("📦 Anomalies file size (MB):", round(os.path.getsize("sherbrooke_sensor_readings_with_anomalies.csv") / 1e6, 2))

# Preview top lines to confirm it's real CSV
try:
    st.sidebar.code(open("sherbrooke_fixed_sensor_readings.csv", "r").readlines()[0:5])
except Exception as e:
    st.sidebar.error(f"Normal CSV Read Error: {e}")

try:
    st.sidebar.code(open("sherbrooke_sensor_readings_with_anomalies.csv", "r").readlines()[0:5])
except Exception as e:
    st.sidebar.error(f"Anomalies CSV Read Error: {e}")




# Sidebar info block

with st.sidebar.expander("📄 Dashboard Info", expanded=False):
    st.markdown("""
    ### ℹ️ About This Dashboard

    This interactive dashboard is designed to:

    - Monitor real-time environmental sensor data  
    - Analyze gas levels across different timeframes (monthly, seasonal, day vs night)  
    - Visualize temperature, humidity, moisture, and gas trends  
    - Explore correlations between gas concentration and environmental variables  
    - Compare gas levels across sensor locations  
    - Detect and observe anomalies in environmental behavior  
    - Drill down into selected timeframes for detailed insights

    ---

    ### Dataset Information
    The data displayed in this dashboard is **synthetic** — generated to replicate realistic environmental conditions.  
    It is intended solely for **research, analysis, and system development** purposes.

    ---

    ### 🛠️ Technologies Used
    - **Python** (data processing & analysis)  
    - **Pandas & NumPy** (data manipulation)  
    - **Matplotlib & Seaborn** (data visualization)  
    - **Streamlit** (interactive UI)  

    ---

    ### Credits
    Developed by: *Jibril Hussaini*
    Supervised by: *Rachid Hedjam*   
    Institution: *Bishop’s University*  
    Year: 2025

    ---

    ### 📤 Future Add-ons
    - PDF/Excel report generation  
    - User-uploaded sensor data  
    - AI-powered gas prediction models  
    - Real-time map-based visualization
    """)


# 📌 Sidebar Option: Show Summary Statistics
show_summary = st.sidebar.checkbox(" Show Statistical Summary")

# -------------------- Statistical Summary Section --------------------
if show_summary:
    st.markdown("## Statistical Summary of Selected Dataset")

    # Compute full descriptive statistics
    summary = data[["Temperature", "Humidity", "Moisture", "Gas"]].describe().T

    # Optionally, you can round values for display
    summary = summary.round(2)

    # Rename for clarity (optional)
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

    # Display as styled dataframe
    st.dataframe(summary.style.format("{:.2f}"))


# Convert the selected data (normal or anomalies) to CSV format
csv = data.to_csv(index=False)
csv_bytes = csv.encode('utf-8')

# Download button in sidebar
with st.sidebar.expander("📥 Download Reports", expanded=False):
    st.markdown("You can download the current dataset as a CSV file.")
    st.download_button(
        label="⬇️ Download CSV Report",
        data=csv_bytes,
        file_name='sensor_data_report.csv',
        mime='text/csv'
    )


# ----------------------- Sensor Overview Section -----------------------

import datetime  # Make sure this is at the top of your script

# -------------------- Real-Time Sensor Overview (Manual Refresh Only) --------------------

st.markdown("## 🌡️ Real-Time Sensor Overview")

# Initialize with a sample row (only once)
if 'random_row' not in st.session_state:
    if not data.empty:
        st.session_state.random_row = data.sample(1).iloc[0]
        st.session_state.last_update = datetime.now()
    else:
        st.warning("⚠️ No data available to sample.")
        st.stop()

# Manual Refresh Button
if st.button("🔁 Refresh Sensor Data"):
    if not data.empty:
        st.session_state.random_row = data.sample(1).iloc[0]
        st.session_state.last_update = datetime.now()
    else:
        st.warning("⚠️ No data available to refresh.")
        st.stop()


# Display last update timestamp
if 'last_update' in st.session_state:
    st.caption(f"Last updated: {st.session_state.last_update.strftime('%Y-%m-%d %I:%M:%S %p')}")


# Display metrics
random_row = st.session_state.random_row
cols = st.columns(4)

cols[0].metric(label=f"Temperature: {random_row['Location']}", value=f"{round(random_row['Temperature'], 2)} °C", delta="Last update")
cols[1].metric(label=f"Humidity: {random_row['Location']}", value=f"{round(random_row['Humidity'], 2)} %", delta="Last update")
cols[2].metric(label=f"Moisture: {random_row['Location']}", value=f"{round(random_row['Moisture'], 2)}", delta="Last update")
cols[3].metric(label=f"Gas: {random_row['Location']}", value=f"{round(random_row['Gas'], 2)}", delta="Last update")




# Add a dropdown to choose the type of gas level trend
plot_option = st.selectbox("📈 Select Gas Level Trend View:", 
    ["Select an option", 
     "Seasonal Average", 
     "Monthly Trend", 
     "Day vs Night Gas Levels",  # ✅ Already present
     "Sensor-wise Comparison"]  # ✅ This was missing before
)

# ➤ 1. Seasonal Trend
if plot_option == "Seasonal Average":
    data['Season'] = data.index.month.map({
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Fall", 10: "Fall", 11: "Fall"
    })

    seasonal_trends = data.groupby('Season')['Gas'].mean().reindex(["Spring", "Summer", "Fall", "Winter"])

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=seasonal_trends.index, y=seasonal_trends.values, palette="coolwarm", ax=ax, legend=False)
    ax.set_xlabel("Season")
    ax.set_ylabel("Average Gas Level")
    ax.set_title("Average Gas Levels Across Seasons")
    st.pyplot(fig)

# ➤ 2. Monthly Trend
elif plot_option == "Monthly Trend":
    monthly_trends = data.groupby(data.index.month)['Gas'].mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x=monthly_trends.index, y=monthly_trends.values, marker="o", ax=ax)
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Gas Level")
    ax.set_title("Monthly Gas Level Trends")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.grid(True)
    st.pyplot(fig)

# ➤ 3. Day vs Night Comparison
elif plot_option == "Day vs Night Gas Levels":
    # Extract hour from datetime index
    data['Hour'] = data.index.hour
    data['TimeOfDay'] = data['Hour'].apply(lambda x: 'Day (6AM-6PM)' if 6 <= x < 18 else 'Night (6PM-6AM)')

    # Calculate average gas levels for day vs night
    day_night_avg = data.groupby('TimeOfDay')['Gas'].mean().reindex(['Day (6AM-6PM)', 'Night (6PM-6AM)'])

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=day_night_avg.index, y=day_night_avg.values, palette="Set2", ax=ax)
    ax.set_ylabel("Average Gas Level")
    ax.set_title("Gas Levels During Day vs Night")
    st.pyplot(fig)


# ➤ Sensor-wise Gas Comparison
elif plot_option == "Sensor-wise Comparison":
    top_n = 20
    sensor_avg = data.groupby('Location')['Gas'].mean().sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = sns.barplot(x=sensor_avg.values, y=sensor_avg.index, palette="viridis", ax=ax)

    # Add value labels with smaller offset
    for i, (value, label) in enumerate(zip(sensor_avg.values, sensor_avg.index)):
        ax.text(value + 0.3, i, f"{value:.2f}", va='center', fontsize=9)

    ax.set_xlabel("Average Gas Level")
    ax.set_ylabel("Sensor Location")
    ax.set_title(f"Top {top_n} Sensors by Average Gas Levels")

    # Adjust layout to avoid clipping
    plt.tight_layout()
    st.pyplot(fig)




# ➤ Default message
else:
    st.info("ℹ️ Please select a trend type to display the chart.")






# -------------------- Environmental Insights Section --------------------

st.markdown("## 🌍 Environmental Insights View")

plot_env_option = st.selectbox("📊 Select Environmental View Type:", 
                               ["Select an option", 
                                "Monthly Trends of All Variables", 
                                "Seasonal Trends of Environmental Variables", 
                                "Correlation Matrix (Main Vars)", 
                                "Full Correlation Matrix (All Vars)"])

# ➤ 1. Monthly Trends of All Variables
if plot_env_option == "Monthly Trends of All Variables":
    monthly_avg = data.groupby(data.index.month)[["Temperature", "Humidity", "Moisture", "Gas"]].mean()

    fig, ax = plt.subplots(figsize=(12, 6))
    for column in monthly_avg.columns:
        ax.plot(monthly_avg.index, monthly_avg[column], marker="o", label=column)

    ax.set_title("Monthly Trends of Temperature, Humidity, Moisture & Gas")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Value")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

# ➤ 2. Seasonal Trends of Environmental Variables
elif plot_env_option == "Seasonal Trends of Environmental Variables":
    data['Season'] = data.index.month.map({
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Fall", 10: "Fall", 11: "Fall"
    })

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    sns.barplot(x='Season', y='Temperature', data=data, ax=axs[0],
                order=["Spring", "Summer", "Fall", "Winter"], palette="coolwarm")
    axs[0].set_title("Temperature by Season")

    sns.barplot(x='Season', y='Humidity', data=data, ax=axs[1],
                order=["Spring", "Summer", "Fall", "Winter"], palette="Blues")
    axs[1].set_title("Humidity by Season")

    sns.barplot(x='Season', y='Moisture', data=data, ax=axs[2],
                order=["Spring", "Summer", "Fall", "Winter"], palette="Greens")
    axs[2].set_title("Moisture by Season")

    st.pyplot(fig)

# ➤ 3. Main Correlation Matrix
elif plot_env_option == "Correlation Matrix (Main Vars)":
    corr = data[["Temperature", "Humidity", "Moisture", "Gas"]].corr()

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True, ax=ax)
    ax.set_title("Correlation Matrix of Temperature, Humidity, Moisture & Gas")
    st.pyplot(fig)

# ➤ 4. Full Correlation Matrix with Time-based Features
elif plot_env_option == "Full Correlation Matrix (All Vars)":
    df_corr = data.copy()

    # Add time-based columns if not present
    if 'Hour' not in df_corr.columns:
        df_corr["Hour"] = df_corr.index.hour
    if 'DayOfWeek' not in df_corr.columns:
        df_corr["DayOfWeek"] = df_corr.index.dayofweek
    if 'Month' not in df_corr.columns:
        df_corr["Month"] = df_corr.index.month

    # Compute correlation matrix for all numeric columns
    corr_matrix = df_corr.select_dtypes(include=['number']).corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
    ax.set_title('Correlation Between Gas Leaks and Environmental Variables')
    st.pyplot(fig)

# ➤ Default message
elif plot_env_option == "Select an option":
    st.info("ℹ️ Please select an environmental insight view.")




# -------------------- Time Series Monitoring Section --------------------
st.markdown("## 📉 Time Series Monitoring with Summary Stats")

# Step 1: Select variable
variable_to_plot = st.selectbox("📌 Select a variable to monitor over time:", 
                                ["Temperature", "Humidity", "Moisture", "Gas"])

# Step 2: Choose view mode
view_mode = st.radio("⏱️ View data by:", ["Daily", "Weekly", "Monthly", "Yearly"], horizontal=True)

# Step 3: Filter data
if view_mode == "Daily":
    selected_date = st.date_input("📅 Select a day", value=datetime.datetime(2023, 1, 1))
    daily_data = data[data.index.date == selected_date]
    title = f"{variable_to_plot} - {selected_date.strftime('%B %d, %Y')} (Daily View)"
    resample_freq = "H"  # hourly
    filtered = daily_data

elif view_mode == "Weekly":
    selected_week = st.date_input("📅 Select any date in the week", value=datetime.datetime(2023, 1, 1))
    start_of_week = selected_week - timedelta(days=selected_week.weekday())
    end_of_week = start_of_week + timedelta(days=6)
    weekly_data = data[(data.index.date >= start_of_week) & (data.index.date <= end_of_week)]
    title = f"{variable_to_plot} - Week of {start_of_week.strftime('%b %d')} (Weekly View)"
    resample_freq = "6H"
    filtered = weekly_data

elif view_mode == "Monthly":
    month_map = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12
    }
    selected_month = st.selectbox("📆 Select month:", list(month_map.keys()))
    month_num = month_map[selected_month]
    monthly_data = data[data.index.month == month_num]
    title = f"{variable_to_plot} - {selected_month} (Monthly View)"
    resample_freq = "D"
    filtered = monthly_data

elif view_mode == "Yearly":
    selected_year = st.selectbox("📅 Select year:", sorted(data.index.year.unique(), reverse=True))
    yearly_data = data[data.index.year == selected_year]
    title = f"{variable_to_plot} - {selected_year} (Yearly View)"
    resample_freq = "M"
    filtered = yearly_data

# Step 4: Plot if not empty
if not filtered.empty:
    ts_data = filtered[[variable_to_plot]].resample(resample_freq).mean().dropna()

    min_val = ts_data[variable_to_plot].min()
    max_val = ts_data[variable_to_plot].max()
    avg_val = ts_data[variable_to_plot].mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ts_data.index, ts_data[variable_to_plot], color='steelblue', linewidth=2)
    ax.fill_between(ts_data.index, ts_data[variable_to_plot], color='skyblue', alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel(variable_to_plot)
    ax.grid(True)

    # Format x-axis as Day/Month
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))

    st.pyplot(fig)

    # Summary at the bottom
    with st.container():
        col1, col2, col3, _ = st.columns([1, 1, 1, 6])
        col1.markdown(f"<div style='text-align:center;'><span style='color:#FF5733; font-weight:bold;'>min</span><br>{round(min_val, 2)}</div>", unsafe_allow_html=True)
        col2.markdown(f"<div style='text-align:center;'><span style='color:#FF5733; font-weight:bold;'>max</span><br>{round(max_val, 2)}</div>", unsafe_allow_html=True)
        col3.markdown(f"<div style='text-align:center;'><span style='color:#FF5733; font-weight:bold;'>avg</span><br>{round(avg_val, 2)}</div>", unsafe_allow_html=True)
else:
    st.warning("⚠️ No data found for the selected time range.")


