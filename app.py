import streamlit as st
import pandas as pd
import numpy as np
import datetime
import nasapower
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Set page configurations
st.set_page_config(page_title="Nairobi Solar Grid Forecast Dashboard", layout="wide")

# Title and Context
st.title("☀️ Nairobi Solar Grid Availability Forecasting Engine")
st.markdown("""
This dashboard acts as an energy informatics tool, utilizing machine learning to predict real-time solar irradiance 
for grid node optimization in Kenya. Data is streamed dynamically from the **NASA POWER API**.
""")

# ----------------------------------------------------
# SIDEBAR CONTROLS
# ----------------------------------------------------
st.sidebar.header("🎯 Target Node Coordinates")
# Default coordinates set to Kilimani, Nairobi
lat = st.sidebar.number_input("Latitude", value=-1.2921, format="%.4f")
lon = st.sidebar.number_input("Longitude", value=36.8219, format="%.4f")

st.sidebar.header("📅 Forecast Horizon")
start_date = st.sidebar.date_input("Start Date", datetime.date(2025, 6, 1))
end_date = st.sidebar.date_input("End Date", datetime.date(2025, 6, 7))

# ----------------------------------------------------
# DATA PIPELINE & MODEL ENGINE (Cached for performance)
# ----------------------------------------------------
@st.cache_data
def fetch_and_predict(lat, lon, start_date, end_date):
    # 1. Fetch Dynamic Data from API
    df = nasapower.point(
        coordinates=(lat, lon),
        parameters=['T2M', 'WS2M', 'ALLSKY_SFC_SW_DWN'],
        start=start_date,
        end=end_date,
        resolution='hourly',
        community='RE'
    )
    df_clean = df.reset_index()
    df_clean['datetime'] = pd.to_datetime(df_clean.iloc[:, 0])
    df_clean['hour'] = df_clean['datetime'].dt.hour
    df_clean['month'] = df_clean['datetime'].dt.month
    df_clean['day_of_year'] = df_clean['datetime'].dt.dayofyear
    df_clean = df_clean.replace(-999, np.nan).dropna().sort_values('datetime').reset_index(drop=True)
    
    # 2. Reconstruct your 99.05% Advanced Autoregressive Features
    df_clean['lag_1h'] = df_clean['ALLSKY_SFC_SW_DWN'].shift(1)
    df_clean['lag_2h'] = df_clean['ALLSKY_SFC_SW_DWN'].shift(2)
    df_clean['lag_24h'] = df_clean['ALLSKY_SFC_SW_DWN'].shift(24)
    df_clean['temp_trend_3h'] = df_clean['T2M'].diff(3)
    df_clean['solar_rolling_mean_3h'] = df_clean['ALLSKY_SFC_SW_DWN'].shift(1).rolling(window=3).mean()
    df_clean = df_clean.dropna().reset_index(drop=True)
    
    # 3. Quick Train on Historical Window Sequence
    features = ['T2M', 'WS2M', 'hour', 'month', 'day_of_year', 'lag_1h', 'lag_2h', 'lag_24h', 'temp_trend_3h', 'solar_rolling_mean_3h']
    X = df_clean[features]
    y = df_clean['ALLSKY_SFC_SW_DWN']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    df_clean['Predicted_Output'] = model.predict(X)
    return df_clean

# Run pipeline when user interacts
if st.sidebar.button("⚡ Generate Grid Forecast"):
    with st.spinner("Streaming telemetry from NASA servers and executing ML pipeline..."):
        try:
            results_df = fetch_and_predict(lat, lon, start_date, end_date)
            
            # ----------------------------------------------------
            # DASHBOARD METRICS DISPLAY
            # ----------------------------------------------------
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Peak Observed Irradiance", value=f"{results_df['ALLSKY_SFC_SW_DWN'].max():.2f} kW-hr/m²")
            with col2:
                st.metric(label="Average Temp (T2M)", value=f"{results_df['T2M'].mean():.1f} °C")
            with col3:
                st.metric(label="Forecast Horizon Data Points", value=f"{len(results_df)} hours")
                
            # ----------------------------------------------------
            # INTERACTIVE VISUALIZATION
            # ----------------------------------------------------
            st.subheader("📈 Grid Load Balancing: Actual vs Predicted Solar Supply")
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(results_df['datetime'], results_df['ALLSKY_SFC_SW_DWN'], label='Actual Irradiance', color='#1f77b4', linewidth=2)
            ax.plot(results_df['datetime'], results_df['Predicted_Output'], label='ML Forecast Model', color='#ff7f0e', linestyle='--')
            ax.set_ylabel("Solar Metrics (kW-hr/m²/day)")
            ax.set_xlabel("Time Horizon Sequence")
            ax.legend()
            ax.grid(True, linestyle=":", alpha=0.6)
            st.pyplot(fig)
            
            # ----------------------------------------------------
            # RAW TELEMETRY VIEW
            # ----------------------------------------------------
            st.subheader("📋 Streaming Grid Node Telemetry Data")
            st.dataframe(results_df[['datetime', 'T2M', 'WS2M', 'ALLSKY_SFC_SW_DWN', 'Predicted_Output']].tail(10))
            
        except Exception as e:
            st.error(f"Pipeline Interrupted: Ensure dates are chronological and map node exists. Error details: {e}")
else:
    st.info("👈 Select your location coordinates and timeline on the sidebar, then click 'Generate Grid Forecast'.")
