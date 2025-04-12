import altair as alt

# 🔁 Reload full dataset (not just .head())
df_plot = pd.read_csv('data/sherbrooke_fixed_sensor_readings.csv')

# 🧠 Ensure Datetime is combined properly
df_plot['Datetime'] = pd.to_datetime(df_plot['Date'] + ' ' + df_plot['Time'], errors='coerce')
df_plot.dropna(subset=['Datetime'], inplace=True)

# 🎛️ Dropdown to pick variable to plot
plot_option = st.selectbox("📊 Select variable to visualize:", ['Temperature', 'Humidity', 'Moisture', 'Gas'])

# 📈 Create interactive chart
chart = alt.Chart(df_plot).mark_line(interpolate='monotone').encode(
    x=alt.X('Datetime:T', title='Datetime'),
    y=alt.Y(f'{plot_option}:Q', title=plot_option),
    tooltip=['Datetime:T', f'{plot_option}:Q', 'Location', 'Gas_Level']
).properties(
    title=f"{plot_option} Over Time",
    width=1000,
    height=400
).interactive()

# 🔍 Display chart
st.altair_chart(chart, use_container_width=True)
