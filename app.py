import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import pickle
from datetime import datetime
import openmeteo_requests
from retry_requests import retry
import requests_cache
import matplotlib.pyplot as plt
import shap

st.set_page_config(page_title="Smart Grid Forecaster", layout="wide", page_icon="⚡")

st.title("⚡ Smart Grid Energy Forecaster")
st.caption("**Top 1% Capstone Project** — 24-Hour Ahead Prediction with Live Weather, Quantile Regression & Live SHAP Explainability")

# Sidebar - Household Profile
with st.sidebar:
    st.header("🏠 Household Profile")
    num_people = st.number_input("Number of residents", min_value=1, max_value=20, value=4)
    house_size = st.number_input("House size (m²)", min_value=50, max_value=500, value=120)
    has_ac = st.checkbox("Air Conditioners / Cooling", value=True)
    has_solar = st.checkbox("Rooftop Solar Panels", value=False)
    solar_kw = st.number_input("Solar capacity (kW)", min_value=0.0, max_value=20.0, value=5.0) if has_solar else 0.0

    if st.button("🔮 Generate 24-Hour Forecast"):
        st.session_state.run_forecast = True

# Load models (using lightweight SHAP explainer)
@st.cache_resource
def load_models():
    try:
        xgb_model = joblib.load('xgb_model.pkl')
        quantile_models = joblib.load('quantile_models.pkl')
        explainer = joblib.load('shap_explainer_live.pkl')   # ← Use the small one you created
        with open('feature_cols.pkl', 'rb') as f:
            feature_cols = pickle.load(f)
        return xgb_model, quantile_models, explainer, feature_cols
    except Exception as e:
        st.error(f"Could not load models: {e}")
        return None, None, None, None

xgb_model, quantile_models, explainer, feature_cols = load_models()

# Helper functions (include them here so app is self-contained)
def engineer_features(df):
    df = df.copy()
    target = 'energy_kwh'
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    for col, period in [('hour', 24), ('dayofweek', 7), ('month', 12)]:
        df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / period)
        df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / period)
    
    df['temp_lag1'] = df['temperature_2m'].shift(1)
    df['hdd'] = np.maximum(18.0 - df['temperature_2m'], 0)
    df['cdd'] = np.maximum(df['temperature_2m'] - 18.0, 0)
    df['cloud_impact'] = df['cloud_cover'] / 100.0
    df['effective_radiation'] = df['shortwave_radiation'] * (1 - df['cloud_impact'])
    
    lags = [1, 2, 3, 6, 12, 24, 48, 168]
    for lag in lags:
        df[f'lag_{lag}h'] = df[target].shift(lag)
    
    windows = [6, 12, 24, 48]
    for w in windows:
        df[f'roll_mean_{w}h'] = df[target].shift(1).rolling(w).mean()
        df[f'roll_std_{w}h'] = df[target].shift(1).rolling(w).std()
    
    return df

def generate_smart_schedule(forecast, weather_df):
    hours = [datetime.now().replace(minute=0, second=0) + pd.Timedelta(hours=i) for i in range(24)]
    data = []
    for i, (dt, demand) in enumerate(zip(hours, forecast)):
        h = dt.hour
        price = 9 if h < 6 else 15 if h < 9 else 12 if h < 16 else 28 if h < 20 else 20 if h < 22 else 11
        solar = 0
        if 'shortwave_radiation' in weather_df.columns:
            rad = weather_df['shortwave_radiation'].iloc[i] if i < len(weather_df) else 0
            solar = min(3.5, rad / 1000 * 3.5 * 0.75)
        net = max(0, demand - solar)
        rec = "✅ Run heavy appliances" if price <= 11 else "☀️ Maximize solar use" if solar > 1.0 else "❄️ High AC demand" if weather_df['temperature_2m'].iloc[i] > 28 else "Normal"
        
        # FIX APPLIED HERE: Added 'Price_c/kWh': price to the dictionary
        data.append({
            'Hour': dt.strftime('%H:%M'), 
            'Demand_kWh': round(demand,2), 
            'Solar_kWh': round(solar,2), 
            'Net_kWh': round(net,2), 
            'Price_c/kWh': price, 
            'Recommendation': rec
        })
    return pd.DataFrame(data)

# Main Forecast
if 'run_forecast' in st.session_state and st.session_state.run_forecast:
    with st.spinner("Fetching live weather and computing forecast..."):
        # Weather
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        openmeteo = openmeteo_requests.Client(session=retry(cache_session, retries=5))
        
        params = {
            "latitude": 18.62, "longitude": 73.80,
            "hourly": ["temperature_2m", "cloud_cover", "shortwave_radiation", "wind_speed_10m"],
            "forecast_days": 2, "timezone": "Asia/Kolkata"
        }
        
        response = openmeteo.weather_api("https://api.open-meteo.com/v1/forecast", params=params)[0]
        hourly = response.Hourly()
        time_index = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive='left'
        ).tz_convert("Asia/Kolkata").tz_localize(None)
        
        weather_data = {"datetime": time_index}
        for i, var in enumerate(["temperature_2m", "cloud_cover", "shortwave_radiation", "wind_speed_10m"]):
            weather_data[var] = hourly.Variables(i).ValuesAsNumpy()[:len(time_index)]
        
        weather_df = pd.DataFrame(weather_data).set_index('datetime')
        next_24 = weather_df.head(24).copy()
        
        # Feature Engineering
        dummy = pd.DataFrame(index=next_24.index).join(next_24)
        for col in ['Global_active_power','Global_reactive_power','Voltage','Global_intensity',
                    'Sub_metering_1','Sub_metering_2','Sub_metering_3','relative_humidity_2m','precipitation']:
            dummy[col] = 0.0
        dummy['energy_kwh'] = 0.0
        
        user_feat_df = engineer_features(dummy)
        user_feat = user_feat_df[[c for c in feature_cols if c in user_feat_df.columns]].copy().astype(np.float32)
        for c in feature_cols:
            if c not in user_feat.columns:
                user_feat[c] = 0.0
        user_feat = user_feat[feature_cols].astype(np.float32)
        
        # Scaling & Prediction
        scale = (num_people / 4.0) * (house_size / 120.0) * (1.45 if has_ac else 1.0)
        lower = quantile_models[0.05].predict(user_feat) * scale
        median = quantile_models[0.50].predict(user_feat) * scale
        upper = quantile_models[0.95].predict(user_feat) * scale
        
        if has_solar:
            solar_est = (next_24['shortwave_radiation'].values / 1000) * solar_kw * 0.78
            median = np.maximum(0, median - solar_est)
            lower = np.maximum(0, lower - solar_est*0.9)
            upper = np.maximum(0, upper - solar_est*1.1)
        
        final_pred = np.clip(median, 0, None)
        schedule = generate_smart_schedule(final_pred, next_24)
        
        daily_cost = (schedule['Net_kWh'] * schedule['Price_c/kWh'] / 100 * 83).sum()
        
        st.success(f"✅ Forecast Complete! Total expected consumption: **{final_pred.sum():.1f} kWh**")
        
        # Main Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(24)), y=final_pred, mode='lines+markers', name='Predicted Demand', line=dict(color='#1e88e5', width=4)))
        fig.add_trace(go.Scatter(x=list(range(24)), y=upper, mode='lines', name='Upper 90% CI', line=dict(color='rgba(233,30,99,0.5)', dash='dash')))
        fig.add_trace(go.Scatter(x=list(range(24)), y=lower, mode='lines', name='Lower 90% CI', line=dict(color='rgba(233,30,99,0.5)', dash='dash'), fill='tonexty'))
        fig.add_trace(go.Bar(x=list(range(24)), y=schedule['Solar_kWh'], name='Solar Generation', marker_color='#fdd835'))
        fig.update_layout(title="24-Hour Energy Demand Forecast with 90% Confidence Interval", xaxis_title="Hour Ahead", yaxis_title="kWh", height=650)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Smart Grid Schedule")
        # Added the new Price column to your dataframe view so you can see it working!
        st.dataframe(schedule[['Hour', 'Demand_kWh', 'Solar_kWh', 'Net_kWh', 'Price_c/kWh', 'Recommendation']].round(2), use_container_width=True)
        
        st.metric("Estimated Daily Cost (Smart Schedule)", f"₹{daily_cost:.0f}")

# Live SHAP Section
st.subheader("🔍 Live SHAP Explainability")
if st.button("Compute Live SHAP for this forecast"):
    with st.spinner("Computing SHAP values..."):
        sample_feat = user_feat[:6] if 'user_feat' in locals() else None
        if sample_feat is not None and explainer is not None:
            shap_values = explainer.shap_values(sample_feat)
            fig, ax = plt.subplots(figsize=(12, 8))
            shap.summary_plot(shap_values, sample_feat, feature_names=feature_cols, show=False)
            plt.title("Live SHAP Feature Impact")
            st.pyplot(fig)
        else:
            st.warning("Run forecast first or check explainer.")

st.caption("Capstone Project | Live Weather + Quantile Regression + Live SHAP")
