import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import pickle
import requests
from datetime import datetime
import openmeteo_requests
from retry_requests import retry
import requests_cache
import matplotlib.pyplot as plt
import shap

# 1. PAGE CONFIGURATION (Must be first)
st.set_page_config(page_title="Smart Grid Forecaster", layout="wide", page_icon="⚡")

# --- CSS Styling for a cleaner look ---
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1 { color: #1E88E5; }
    </style>
""", unsafe_allow_html=True)

# 2. HEADER
st.title("⚡ Smart Grid Energy Forecaster")
st.caption("**Top 1% Capstone Project** — 24-Hour Ahead Prediction with Live Weather, Quantile Regression & Live SHAP")

# 3. SIDEBAR: PROFESSIONAL INPUTS
with st.sidebar:
    st.header("🌍 Location Settings")
    city_input = st.text_input("Enter City Name", value="Pune", placeholder="e.g., London, New York, Tokyo")
    
    st.divider()
    
    st.header("🏠 Household Profile")
    num_people = st.number_input("Number of residents", min_value=1, max_value=20, value=4)
    house_size = st.number_input("House size (m²)", min_value=50, max_value=500, value=120, step=10)
    has_ac = st.checkbox("Air Conditioners / Cooling", value=True)
    has_solar = st.checkbox("Rooftop Solar Panels", value=False)
    solar_kw = st.number_input("Solar capacity (kW)", min_value=0.0, max_value=20.0, value=5.0) if has_solar else 0.0

    st.divider()
    if st.button("🔮 Generate 24-Hour Forecast", use_container_width=True, type="primary"):
        st.session_state.run_forecast = True

# 4. HELPER FUNCTIONS
@st.cache_data(ttl=3600)
def get_coordinates(city_name):
    """Fetch lat/lon for a given city using Open-Meteo's free Geocoding API."""
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1&format=json"
    try:
        response = requests.get(url).json()
        if "results" in response and len(response["results"]) > 0:
            loc = response["results"][0]
            return loc["latitude"], loc["longitude"], loc["name"], loc.get("country", "")
    except Exception as e:
        return None, None, None, None
    return None, None, None, None

@st.cache_resource
def load_models():
    """Load pre-trained models and feature columns."""
    try:
        xgb_model = joblib.load('xgb_model.pkl')
        quantile_models = joblib.load('quantile_models.pkl')
        explainer = joblib.load('shap_explainer_live.pkl')
        with open('feature_cols.pkl', 'rb') as f:
            feature_cols = pickle.load(f)
        return xgb_model, quantile_models, explainer, feature_cols
    except Exception as e:
        st.error(f"⚠️ Could not load models. Please check your files. Error: {e}")
        return None, None, None, None

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
        rec = "✅ Run appliances" if price <= 11 else "☀️ Maximize solar" if solar > 1.0 else "❄️ High AC risk" if weather_df['temperature_2m'].iloc[i] > 28 else "Normal"
        
        data.append({
            'Hour': dt.strftime('%H:%M'), 
            'Demand_kWh': round(demand, 2), 
            'Solar_kWh': round(solar, 2), 
            'Net_kWh': round(net, 2), 
            'Price_c/kWh': price, 
            'Recommendation': rec
        })
    return pd.DataFrame(data)

# 5. MAIN EXECUTION
xgb_model, quantile_models, explainer, feature_cols = load_models()

if 'run_forecast' in st.session_state and st.session_state.run_forecast:
    
    # --- Step A: Resolve Location ---
    lat, lon, resolved_city, country = get_coordinates(city_input)
    
    if lat is None:
        st.error(f"❌ Could not find coordinates for '{city_input}'. Please check the spelling.")
    else:
        with st.spinner(f"📡 Fetching live weather for {resolved_city}, {country} and computing forecast..."):
            
            # --- Step B: Weather API (Using Dynamic Lat/Lon) ---
            cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
            openmeteo = openmeteo_requests.Client(session=retry(cache_session, retries=5))
            
            params = {
                "latitude": lat, 
                "longitude": lon,
                "hourly": ["temperature_2m", "cloud_cover", "shortwave_radiation", "wind_speed_10m"],
                "forecast_days": 2, 
                "timezone": "auto" # Automatically adjusts to the city's timezone!
            }
            
            response = openmeteo.weather_api("https://api.open-meteo.com/v1/forecast", params=params)[0]
            hourly = response.Hourly()
            
            # Use UTC and convert safely
            time_index = pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive='left'
            )
            
            weather_data = {"datetime": time_index}
            for i, var in enumerate(["temperature_2m", "cloud_cover", "shortwave_radiation", "wind_speed_10m"]):
                weather_data[var] = hourly.Variables(i).ValuesAsNumpy()[:len(time_index)]
            
            weather_df = pd.DataFrame(weather_data).set_index('datetime')
            next_24 = weather_df.head(24).copy()
            
            # --- Step C: Feature Engineering ---
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
            
            # --- Step D: Scaling & Prediction ---
            scale = (num_people / 4.0) * (house_size / 120.0) * (1.45 if has_ac else 1.0)
            lower = quantile_models[0.05].predict(user_feat) * scale
            median = quantile_models[0.50].predict(user_feat) * scale
            upper = quantile_models[0.95].predict(user_feat) * scale
            
            total_solar_gen = 0
            if has_solar:
                solar_est = (next_24['shortwave_radiation'].values / 1000) * solar_kw * 0.78
                total_solar_gen = solar_est.sum()
                median = np.maximum(0, median - solar_est)
                lower = np.maximum(0, lower - solar_est*0.9)
                upper = np.maximum(0, upper - solar_est*1.1)
            
            final_pred = np.clip(median, 0, None)
            schedule = generate_smart_schedule(final_pred, next_24)
            
            daily_cost = (schedule['Net_kWh'] * schedule['Price_c/kWh'] / 100 * 83).sum()
            
            # --- Step E: Professional Dashboard Layout ---
            st.success(f"✅ Live forecast generated for **{resolved_city}, {country}**")
            
            # Top KPI Metrics Cards
            col1, col2, col3 = st.columns(3)
            col1.metric(label="Total Expected Demand", value=f"{final_pred.sum():.1f} kWh")
            col2.metric(label="Expected Solar Generation", value=f"{total_solar_gen:.1f} kWh" if has_solar else "0.0 kWh")
            col3.metric(label="Estimated Daily Cost", value=f"₹{daily_cost:.0f}")

            # Main Plot
            st.markdown("### 📈 24-Hour Energy Demand Profile")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(24)), y=final_pred, mode='lines+markers', name='Net Demand', line=dict(color='#1e88e5', width=4)))
            fig.add_trace(go.Scatter(x=list(range(24)), y=upper, mode='lines', name='Upper 90% CI', line=dict(color='rgba(233,30,99,0.5)', dash='dash')))
            fig.add_trace(go.Scatter(x=list(range(24)), y=lower, mode='lines', name='Lower 90% CI', line=dict(color='rgba(233,30,99,0.5)', dash='dash'), fill='tonexty'))
            if has_solar:
                fig.add_trace(go.Bar(x=list(range(24)), y=schedule['Solar_kWh'], name='Solar Gen', marker_color='#fdd835', opacity=0.6))
            
            fig.update_layout(xaxis_title="Hours Ahead", yaxis_title="Energy (kWh)", height=500, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
            # Collapsible Data Table
            with st.expander("📅 View Detailed Smart Grid Schedule & Pricing"):
                st.dataframe(
                    schedule[['Hour', 'Demand_kWh', 'Solar_kWh', 'Net_kWh', 'Price_c/kWh', 'Recommendation']].round(2), 
                    use_container_width=True,
                    hide_index=True
                )

# 6. SHAP EXPLAINABILITY (Bottom Section)
st.divider()
st.subheader("🔍 AI Model Explainability (SHAP)")
st.markdown("Understand *why* the AI made these predictions based on real-time weather and temporal features.")

if st.button("Compute Live Feature Impact"):
    with st.spinner("Analyzing model decisions..."):
        sample_feat = user_feat[:6] if 'user_feat' in locals() else None
        if sample_feat is not None and explainer is not None:
            shap_values = explainer.shap_values(sample_feat)
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values, sample_feat, feature_names=feature_cols, show=False)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("Please generate a forecast first to view SHAP analysis.")
