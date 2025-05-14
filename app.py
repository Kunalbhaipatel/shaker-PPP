
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")
st.title("🛠️ Shaker GPM Capacity vs Performance with Alerts")

st.sidebar.header("Upload CSV")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    st.sidebar.markdown("---")
    max_rows = st.sidebar.slider("Rows to load", 5000, 200000, 30000, step=5000)
    with st.spinner("Loading..."):
        usecols = [
            'YYYY/MM/DD', 'HH:MM:SS',
            'SHAKER #1 (Units)', 'SHAKER #2 (Units)', 'SHAKER #3 (PERCENT)',
            'Total Pump Output (gal_per_min)', 'DAS Vibe Lateral Max (g_force)'
        ]
        df = pd.read_csv(uploaded_file, usecols=usecols, nrows=max_rows)
        df['Timestamp'] = pd.to_datetime(df['YYYY/MM/DD'] + ' ' + df['HH:MM:SS'], format='%m/%d/%Y %H:%M:%S')
        df.set_index('Timestamp', inplace=True)
        df.drop(columns=['YYYY/MM/DD', 'HH:MM:SS'], inplace=True)

    st.success(f"Loaded {len(df)} rows.")

    tab = st.selectbox("View", ["Shaker Trends", "GPM vs Performance + Alerts"])

    if tab == "Shaker Trends":
        st.line_chart(df.iloc[::10][['SHAKER #1 (Units)', 'SHAKER #2 (Units)', 'SHAKER #3 (PERCENT)']])

    elif tab == "GPM vs Performance + Alerts":
        df['Shaker Capacity (GPM)'] = df['Total Pump Output (gal_per_min)'] / 3
        df['Shaker Performance Index'] = (100 - df['DAS Vibe Lateral Max (g_force)'] * 3).clip(0, 100)

        df['Overload Alert'] = (df['Shaker Capacity (GPM)'] > df['Shaker Performance Index']).astype(int)

        st.markdown("### 📈 Capacity vs Performance")
        st.line_chart(df[['Shaker Capacity (GPM)', 'Shaker Performance Index']].iloc[::10])

        st.markdown("### 🚨 Overload Alerts (1 = Exceeded Capacity)")
        st.line_chart(df['Overload Alert'].iloc[::10])

        st.warning(f"⚠️ {df['Overload Alert'].sum()} alert points detected where GPM exceeded shaker tolerance.")

    st.sidebar.download_button("Download Alert CSV", df.to_csv().encode(), "shaker_alerts_output.csv")
