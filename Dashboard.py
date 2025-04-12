import altair as alt

# Optional: allow user to choose variable to plot
plot_option = st.selectbox("📊 Select variable to visualize:", ['Temperature', 'Humidity', 'Moisture', 'Gas'])

# Convert Date & Time to full datetime
df_plot = pd.read_csv('data/sherbrooke_fixed_sensor_readings.csv')  # Load full dataset
df_plot['Datetime'] = pd.to_datetime(df_plot['Date'] + ' ' + df_plot['Time'], errors='coerce')

# Filter out missing datetime if any
df_plot = df_plot.dropna(subset=['Datetime'])

# Build Altair interactive line chart
line_chart = alt.Chart(df_plot).mark_line(interpolate='monotone').encode(
    x=alt.X('Datetime:T', title='Datetime'),
    y=alt.Y(f'{plot_option}:Q', title=plot_option),
    tooltip=['Datetime:T', f'{plot_option}:Q', 'Gas_Level', 'Location']
).properties(
    title=f'{plot_option} Over Time (Interactive)',
    width=900,
    height=400
).interactive()

# Display chart
st.altair_chart(line_chart, use_container_width=True)
