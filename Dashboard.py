import altair as alt
import streamlit as st
import pandas as pd  # 👈 Make sure this is included
import altair as alt
import os


@st.cache_data
def load_data():
    df = pd.read_csv(
        'data/sherbrooke_fixed_sensor_readings.csv',
        nrows=1000,  # 👈 Load only 1000 rows to preview
        parse_dates=[['Date', 'Time']],  # 👈 Combine Date + Time
        on_bad_lines='skip'
    )
    df.rename(columns={'Date_Time': 'Datetime'}, inplace=True)
    return df

df_plot = load_data()

st.markdown("### 📊 Select variable to visualize:")
plot_option = st.selectbox("Select variable to visualize:", ['Temperature', 'Humidity', 'Moisture', 'Gas'])

# ✅ Plot basic Altair chart
import altair as alt

chart = alt.Chart(df_plot).mark_line().encode(
    x='Datetime:T',
    y=plot_option,
    tooltip=['Datetime', plot_option]
).properties(
    width=800,
    height=400
).interactive()

st.altair_chart(chart, use_container_width=True)

