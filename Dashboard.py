import streamlit as st
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from datetime import datetime, timedelta
from datetime import datetime, timedelta, date
import kaggle



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

    try:
        # ✅ Download dataset from Kaggle
        kaggle.api.dataset_download_files(
            'jibrilhussaini/synthetic-sherbrooke-sensor-readings',
            path='data',
            unzip=True
        )

        # ✅ Load both datasets
        df = pd.read_csv('data/sherbrooke_fixed_sensor_readings.csv', on_bad_lines='skip')
        data2 = pd.read_csv('data/sherbrooke_sensor_readings_with_anomalies.csv', on_bad_lines='skip')

        # ✅ Combine Date + Time into Datetime and preprocess both
        for d in [df, data2]:
            if 'Date' in d.columns and 'Time' in d.columns:
                d['Datetime'] = pd.to_datetime(d['Date'] + ' ' + d['Time'], errors='coerce')
                d.drop(columns=['Date', 'Time'], inplace=True)
            elif 'Datetime' not in d.columns:
                st.warning("⚠️ Datetime column missing!")

            d.set_index('Datetime', inplace=True)

            if 'Gas_Level' in d.columns:
                d['Gas_Level'] = d['Gas_Level'].astype('category').cat.codes

            d.dropna(inplace=True)  # ✅ Drop nulls safely inside loop

        return df, data2

    except Exception as e:
        st.error(f"❌ Error loading datasets: {e}")
        return pd.DataFrame(), pd.DataFrame()


# ---------------------------------------------
# 🚀 Load the datasets
# ---------------------------------------------
# 🚀 Load the datasets
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

st.markdown("## 📈 Trend Visualizer")

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

    season_order = ["Spring", "Summer", "Fall", "Winter"]
    seasonal_df = data.groupby("Season")["Gas"].mean().reset_index()
    seasonal_df["Season"] = pd.Categorical(seasonal_df["Season"], categories=season_order, ordered=True)
    seasonal_df = seasonal_df.sort_values("Season")

    chart = alt.Chart(seasonal_df).mark_bar().encode(
        x=alt.X("Season:N", sort=season_order),
        y=alt.Y("Gas:Q", title="Average Gas Level"),
        color=alt.Color("Season:N", scale=alt.Scale(scheme="tableau20")),
        tooltip=["Season", "Gas"]
    ).properties(
        title="Average Gas Levels Across Seasons",
        width=600,
        height=400
    )

    st.altair_chart(chart, use_container_width=True)

# ➤ 2. Monthly Gas Level Trend
elif plot_option == "Monthly Trend":
    monthly_df = data.copy()
    monthly_df["Month"] = monthly_df.index.month
    monthly_avg = monthly_df.groupby("Month")["Gas"].mean().reset_index()
    monthly_avg["MonthName"] = monthly_avg["Month"].apply(lambda x: datetime(2023, x, 1).strftime("%b"))

    chart = alt.Chart(monthly_avg).mark_line(point=True).encode(
        x=alt.X("MonthName:N", sort=list(monthly_avg["MonthName"])),
        y=alt.Y("Gas:Q", title="Average Gas Level"),
        tooltip=["MonthName", "Gas"]
    ).properties(
        title="Monthly Gas Level Trends",
        width=700,
        height=400
    )

    st.altair_chart(chart, use_container_width=True)

# ➤ 3. Day vs Night Gas Levels
elif plot_option == "Day vs Night Gas Levels":
    data["Hour"] = data.index.hour
    data["TimeOfDay"] = data["Hour"].apply(lambda x: "Day (6AM–6PM)" if 6 <= x < 18 else "Night (6PM–6AM)")
    daynight_avg = data.groupby("TimeOfDay")["Gas"].mean().reset_index()

    chart = alt.Chart(daynight_avg).mark_bar().encode(
        x=alt.X("TimeOfDay:N", sort=["Day (6AM–6PM)", "Night (6PM–6AM)"]),
        y=alt.Y("Gas:Q", title="Average Gas Level"),
        tooltip=["TimeOfDay", "Gas"]
    ).properties(
        title="Gas Levels: Day vs Night",
        width=400,
        height=600
    )

    st.altair_chart(chart, use_container_width=True)

# ➤ 4. Sensor-wise Gas Level Comparison
elif plot_option == "Sensor-wise Comparison":
    top_n = 20
    sensor_avg = data.groupby("Location")["Gas"].mean().sort_values(ascending=False).head(top_n).reset_index()

    chart = alt.Chart(sensor_avg).mark_bar().encode(
        y=alt.Y("Location:N", sort="-x", title="Sensor Location"),
        x=alt.X("Gas:Q", title="Average Gas Level"),
        tooltip=["Location", "Gas"]
    ).properties(
        title=f"Top {top_n} Sensor Locations by Gas Level",
        width=700,
        height=600
    )

    st.altair_chart(chart, use_container_width=True)

# ➤ Default Info Message
elif plot_option == "Select an option":
    st.info("ℹ️ Please select a gas trend view to begin visualization.")






# -------------------- Time Series Monitoring Section --------------------
st.markdown("## 📉 Time Series Monitoring with Summary Stats")

# Step 1: Select variable (default is 'Gas')
variable_to_plot = st.selectbox(
    "📌 Select a variable to monitor over time:",
    ["Temperature", "Humidity", "Moisture", "Gas"],
    index=3  # 👈 Index 3 corresponds to 'Gas'
)

# Step 2: Choose view mode
view_mode = st.radio("⏱️ View data by:", ["Daily", "Weekly", "Monthly", "Yearly"], horizontal=True)

# Step 3: Filter data by view
filtered = pd.DataFrame()
title = ""
resample_freq = ""

today = datetime.now().date()

if view_mode == "Daily":
    selected_date = st.date_input("📅 Select a day", value=date(2023, 1, 1))
    filtered = data[data.index.date == selected_date]
    title = f"{variable_to_plot} - {selected_date.strftime('%B %d, %Y')} (Daily View)"
    resample_freq = "H"  # hourly

elif view_mode == "Weekly":
    selected_week = st.date_input("📅 Select any date in the week", value=date(2023, 1, 1))
    start_of_week = selected_week - timedelta(days=selected_week.weekday())
    end_of_week = start_of_week + timedelta(days=6)
    filtered = data[(data.index.date >= start_of_week) & (data.index.date <= end_of_week)]
    title = f"{variable_to_plot} - Week of {start_of_week.strftime('%b %d')} (Weekly View)"
    resample_freq = "6H"

elif view_mode == "Monthly":
    selected_month = st.selectbox("📆 Select month:", list(range(1, 13)))
    filtered = data[data.index.month == selected_month]
    title = f"{variable_to_plot} - Month {selected_month} (Monthly View)"
    resample_freq = "D"

elif view_mode == "Yearly":
    selected_year = st.selectbox("📅 Select year:", sorted(data.index.year.unique(), reverse=True))
    filtered = data[data.index.year == selected_year]
    title = f"{variable_to_plot} - {selected_year} (Yearly View)"
    resample_freq = "M"

# Step 4: Plot + Summary
if not filtered.empty:
    ts_data = filtered[[variable_to_plot]].resample(resample_freq).mean().dropna()
    
    min_val = ts_data[variable_to_plot].min()
    max_val = ts_data[variable_to_plot].max()
    avg_val = ts_data[variable_to_plot].mean()

    # 🎯 Interactive Altair plot
    alt_chart = alt.Chart(ts_data.reset_index()).mark_line(interpolate='monotone').encode(
        x=alt.X('Datetime:T', title='Time'),
        y=alt.Y(variable_to_plot, title=variable_to_plot),
        tooltip=['Datetime', variable_to_plot]
    ).properties(
        title=title,
        width=800,
        height=400
    ).interactive()

    st.altair_chart(alt_chart, use_container_width=True)

    # Refined & tightly attached summary stats block (Min → Avg → Max)
    st.markdown("""
    <style>
    .metric-row {
        display: flex;
        justify-content: center;
        gap: 60px;
        margin-top: -10px; /* 👈 reduce vertical gap to plot */
    }
    .metric-col {
        text-align: center;
        color: red;
    }
    .metric-label {
        font-weight: bold;
    }
    .metric-value {
        font-size: 20px;
    }
    </style>
    
    <div class='metric-row'>
        <div class='metric-col'>
            <div class='metric-label'>Min</div>
            <div class='metric-value'>""" + f"{round(min_val, 2)}" + """</div>
        </div>
        <div class='metric-col'>
            <div class='metric-label'>Average</div>
            <div class='metric-value'>""" + f"{round(avg_val, 2)}" + """</div>
        </div>
        <div class='metric-col'>
            <div class='metric-label'>Max</div>
            <div class='metric-value'>""" + f"{round(max_val, 2)}" + """</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    st.warning("⚠️ No data found for the selected time range.")



# -------------------- Environmental Insights Section --------------------
st.markdown("## 🌍 Environmental Insights View")

plot_env_option = st.selectbox("📊 Select Environmental View Type:", 
                               ["Select an option", 
                                "Monthly Trends of All Variables", 
                                "Seasonal Trends of Environmental Variables", 
                                "Correlation Matrix (Main Vars)", 
                                "Full Correlation Matrix (All Vars)"])


if plot_env_option == "Monthly Trends of All Variables":
    # Step 1: Group and reset index
    monthly_avg = data.groupby(data.index.month)[["Temperature", "Humidity", "Moisture", "Gas"]].mean()
    monthly_avg.index.name = "MonthNum"
    monthly_avg = monthly_avg.reset_index()
    monthly_avg["Month"] = monthly_avg["MonthNum"].apply(lambda x: datetime(2023, x, 1).strftime("%b"))
    
    # ✅ Only melt actual environmental variables
    melted = monthly_avg.melt(
        id_vars=["Month"], 
        value_vars=["Temperature", "Humidity", "Moisture", "Gas"],
        var_name="Variable", 
        value_name="Average"
    )
    
    chart = alt.Chart(melted).mark_line(point=True).encode(
        x=alt.X("Month:N", sort=["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]),
        y="Average:Q",
        color="Variable:N",
        tooltip=["Month", "Variable", "Average"]
    ).properties(
        title="📈 Monthly Trends of Temperature, Humidity, Moisture & Gas",
        width=800,
        height=400
    ).interactive()
    
    st.altair_chart(chart, use_container_width=True)



# ➤ 2. Seasonal Trends of Environmental Variables
if plot_env_option == "Seasonal Trends of Environmental Variables":
    # Let user pick the variable
    var_choice = st.selectbox("📌 Choose variable to view by season:", 
                              ["Temperature", "Humidity", "Moisture", "Gas"], 
                              index=0)

    # Add a 'Season' column if not present
    if 'Season' not in data.columns:
        data['Season'] = data.index.month.map({
            12: "Winter", 1: "Winter", 2: "Winter",
            3: "Spring", 4: "Spring", 5: "Spring",
            6: "Summer", 7: "Summer", 8: "Summer",
            9: "Fall", 10: "Fall", 11: "Fall"
        })

    # Group and calculate seasonal average for selected variable
    seasonal_avg = data.groupby("Season")[var_choice].mean().reset_index()

    # Sort by season order
    season_order = ["Spring", "Summer", "Fall", "Winter"]
    seasonal_avg["Season"] = pd.Categorical(seasonal_avg["Season"], categories=season_order, ordered=True)
    seasonal_avg = seasonal_avg.sort_values("Season")

    # Plot using Altair
    chart = alt.Chart(seasonal_avg).mark_bar(color="steelblue").encode(
        x=alt.X("Season:N", sort=season_order),
        y=alt.Y(f"{var_choice}:Q", title=f"Average {var_choice}"),
        tooltip=["Season", var_choice]
    ).properties(
        title=f"Seasonal Average of {var_choice}",
        width=600,
        height=400
    )

    st.altair_chart(chart, use_container_width=True)


# ➤ 3. Main Correlation Matrix
elif plot_env_option == "Correlation Matrix (Main Vars)":
    corr = data[["Temperature", "Humidity", "Moisture", "Gas"]].corr()

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True, ax=ax)
    ax.set_title("🧩 Correlation Matrix of Main Variables")
    st.pyplot(fig)

# ➤ 4. Full Correlation Matrix with Time-based Features
elif plot_env_option == "Full Correlation Matrix (All Vars)":
    df_corr = data.copy()

    if 'Hour' not in df_corr.columns:
        df_corr["Hour"] = df_corr.index.hour
    if 'DayOfWeek' not in df_corr.columns:
        df_corr["DayOfWeek"] = df_corr.index.dayofweek
    if 'Month' not in df_corr.columns:
        df_corr["Month"] = df_corr.index.month

    corr_matrix = df_corr.select_dtypes(include=['number']).corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
    ax.set_title('🔗 Full Correlation Matrix with Time-based Features')
    st.pyplot(fig)

# ➤ Default message
elif plot_env_option == "Select an option":
    st.info("ℹ️ Please select an environmental insight view.")

