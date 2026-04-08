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
from streamlit_geolocation import streamlit_geolocation
import streamlit.components.v1 as components

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="Smart Grid AI Forecaster", layout="wide", page_icon="⚡", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .metric-card { background-color: #f8f9fa; border-radius: 10px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: transparent; border-radius: 4px 4px 0px 0px; padding-top: 10px; padding-bottom: 10px; font-weight: 600; font-size: 16px; }
    </style>
""", unsafe_allow_html=True)

# 2. HEADER
col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.title("⚡ AI Smart Grid Forecaster & Optimizer")
    st.caption("**Top 1% Capstone** | 3D Earth • Live AQI • Financial ROI • Duck Curve Simulator")
with col2:
    st.image("https://cdn-icons-png.flaticon.com/512/2933/2933116.png", width=60)

# 3. SIDEBAR: PROFESSIONAL INPUTS
with st.sidebar:
    st.header("🌍 Location Settings")
    loc = streamlit_geolocation()
    lat, lon, resolved_city, country = None, None, "Pune", ""
    
    if loc and loc.get('latitude') is not None and loc.get('longitude') is not None:
        lat, lon = loc['latitude'], loc['longitude']
        resolved_city = "Current GPS Location"
        st.success("✅ GPS Locked")
        
        fig_globe = go.Figure(go.Scattergeo(
            lon=[lon], lat=[lat], mode='markers+text', text=["📍 You"], textposition="bottom center",
            marker=dict(size=14, color='#E53935', symbol='circle', line=dict(width=2, color='white'))
        ))
        fig_globe.update_geos(
            projection_type="orthographic", showcoastlines=True, coastlinecolor="#00E5FF",
            showland=True, landcolor="#121212", showocean=True, oceancolor="#0A192F",
            lataxis=dict(showgrid=True, gridcolor="#1E2A38"), lonaxis=dict(showgrid=True, gridcolor="#1E2A38")
        )
        fig_globe.update_layout(height=280, margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_globe, use_container_width=True, config={'displayModeBar': False})
        city_input = st.text_input("City Search (Disabled)", value="Using GPS Coordinates", disabled=True)
    else:
        city_input = st.text_input("Enter City Name", value="Pune")
    
    st.divider()
    st.header("🏠 Live Asset Profile")
    num_people = st.slider("Number of residents", 1, 10, 4)
    house_size = st.number_input("House size (m²)", 50, 500, 120, step=10)
    has_ac = st.toggle("❄️ Air Conditioning", value=True)
    
    st.subheader("☀️ Solar & Storage")
    has_solar = st.toggle("Rooftop Solar", value=True)
    solar_kw = st.slider("Solar Array (kW)", 0.0, 20.0, 5.0) if has_solar else 0.0
    has_battery = st.toggle("🔋 Home Battery System", value=True)
    battery_capacity = st.slider("Battery Size (kWh)", 0.0, 30.0, 10.0) if has_battery else 0.0

    st.divider()
    if st.button("🔮 Run AI Optimization", use_container_width=True, type="primary"):
        st.session_state.fetch_data = True

# 4. HELPER FUNCTIONS
@st.cache_data(ttl=3600)
def get_coordinates(city_name):
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1&format=json"
    try:
        res = requests.get(url).json()
        if "results" in res:
            loc = res["results"][0]
            return loc["latitude"], loc["longitude"], loc["name"], loc.get("country", "")
    except: pass
    return None, None, None, None

@st.cache_resource
def load_models():
    try: return joblib.load('xgb_model.pkl'), joblib.load('quantile_models.pkl'), joblib.load('shap_explainer_live.pkl'), pickle.load(open('feature_cols.pkl', 'rb'))
    except: return None, None, None, None

def engineer_features(df, feature_cols):
    df = df.copy()
    df['hour'], df['dayofweek'], df['month'] = df.index.hour, df.index.dayofweek, df.index.month
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    for col, period in [('hour', 24), ('dayofweek', 7), ('month', 12)]:
        df[f'{col}_sin'], df[f'{col}_cos'] = np.sin(2 * np.pi * df[col] / period), np.cos(2 * np.pi * df[col] / period)
    df['temp_lag1'] = df['temperature_2m'].shift(1).fillna(method='bfill')
    df['hdd'] = np.maximum(18.0 - df['temperature_2m'], 0)
    df['cdd'] = np.maximum(df['temperature_2m'] - 18.0, 0)
    df['cloud_impact'] = df['cloud_cover'] / 100.0
    df['effective_radiation'] = df['shortwave_radiation'] * (1 - df['cloud_impact'])
    target = 'energy_kwh'
    for lag in [1, 2, 3, 6, 12, 24, 48, 168]: df[f'lag_{lag}h'] = 0.0
    for w in [6, 12, 24, 48]: 
        df[f'roll_mean_{w}h'] = 0.0
        df[f'roll_std_{w}h'] = 0.0
    return df[feature_cols].astype(np.float32)

def generate_smart_schedule(forecast, weather_df, solar_kw, batt_cap):
    data = []
    curr_batt = 0.0 
    for i, demand in enumerate(forecast):
        h = i # Hour of forecast
        # Dynamic Time-of-Use Pricing (TOU)
        price = 7.47 if h < 6 else 12.45 if h < 9 else 9.96 if h < 16 else 23.24 if h < 21 else 9.13
        
        rad = weather_df['shortwave_radiation'].iloc[i] if 'shortwave_radiation' in weather_df.columns else 0
        solar = min(solar_kw, rad / 1000 * solar_kw * 0.75) if solar_kw > 0 else 0
        
        net_pre_batt = demand - solar
        charge, discharge = 0, 0
        
        if net_pre_batt < 0: 
            charge = min(abs(net_pre_batt), batt_cap - curr_batt)
            curr_batt += charge
            net_grid = 0
        else: 
            if price >= 15: # Smart Logic: Discharge only when grid is expensive
                discharge = min(net_pre_batt, curr_batt)
                curr_batt -= discharge
            net_grid = net_pre_batt - discharge
            
        data.append({'Hour': f"{h:02d}:00", 'Demand': demand, 'Solar': solar, 'Batt': curr_batt, 'Grid': net_grid, 'Price': price})
    return pd.DataFrame(data)

# 5. MAIN LOGIC
xgb, quantiles, explainer, feature_cols = load_models()

if st.session_state.get('fetch_data', False):
    if lat is None: lat, lon, resolved_city, country = get_coordinates(city_input)
    if lat is not None:
        st.session_state['lat'], st.session_state['lon'] = lat, lon
        st.session_state['location_title'] = f"{resolved_city}, {country}" if country else resolved_city
        
        # Weather API
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,cloud_cover,shortwave_radiation,wind_speed_10m,wind_gusts_10m,relative_humidity_2m&daily=temperature_2m_max,temperature_2m_min,sunrise,sunset&timezone=auto"
        w_res = requests.get(url).json()
        
        w_df = pd.DataFrame(w_res['hourly']).set_index(pd.to_datetime(w_res['hourly']['time'])).head(24)
        st.session_state['weather_df'] = w_df
        st.session_state['today_high'] = w_res['daily']['temperature_2m_max'][0]
        st.session_state['today_low'] = w_res['daily']['temperature_2m_min'][0]
        st.session_state['sunrise'] = w_res['daily']['sunrise'][0][-5:]
        st.session_state['sunset'] = w_res['daily']['sunset'][0][-5:]
        
        # AQI API
        aq_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&current=us_aqi,pm10,pm2_5"
        aq_res = requests.get(aq_url).json()
        st.session_state['aqi'] = aq_res['current']['us_aqi']
        
        # AI Preds
        feats = engineer_features(w_df, feature_cols)
        st.session_state['base_demand'] = quantiles[0.50].predict(feats)
        st.session_state['user_feat'] = feats
        st.session_state['app_ready'] = True
        st.session_state['fetch_data'] = False

if st.session_state.get('app_ready', False):
    scale = (num_people / 4.0) * (house_size / 120.0) * (1.45 if has_ac else 1.0)
    demand = np.clip(st.session_state['base_demand'] * scale, 0, None)
    sched = generate_smart_schedule(demand, st.session_state['weather_df'], solar_kw, battery_capacity)
    
    # Financial Calculations
    baseline_cost = (sched['Demand'] * sched['Price']).sum()
    optimized_cost = (sched['Grid'] * sched['Price']).sum()
    savings = baseline_cost - optimized_cost
    
    # Tabs
    t1, t2, t3, t4 = st.tabs(["📊 Performance", "📡 Telemetry", "💰 ROI & Duck Curve", "🧠 AI Explain"])
    
    with t1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Gross Demand", f"{demand.sum():.1f} kWh")
        c2.metric("Solar Yield", f"{sched['Solar'].sum():.1f} kWh")
        c3.metric("Net Grid Draw", f"{sched['Grid'].sum():.1f} kWh", delta=f"-{(demand.sum()-sched['Grid'].sum()):.1f}", delta_color="inverse")
        c4.metric("Grid Stress", "🔴 High" if sched['Grid'].max() > 5 else "🟢 Low")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sched['Hour'], y=sched['Demand'], name="Baseline", line=dict(dash='dash', color='grey')))
        fig.add_trace(go.Scatter(x=sched['Hour'], y=sched['Grid'], name="Optimized", fill='tozeroy', line=dict(color='#E53935', width=3)))
        if has_solar: fig.add_trace(go.Bar(x=sched['Hour'], y=sched['Solar'], name="Solar", marker_color='#FFD54F', opacity=0.5))
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        st.markdown("### 🛰️ Advanced Environmental Monitoring")
        w1, w2, w3, w4 = st.columns(4)
        w1.metric("Current Temp", f"{st.session_state['weather_df']['temperature_2m'].iloc[0]:.1f}°C", f"H:{st.session_state['today_high']} L:{st.session_state['today_low']}")
        w2.metric("Air Quality (AQI)", st.session_state['aqi'], "🟢 Good" if st.session_state['aqi'] < 50 else "🟠 Fair")
        w3.metric("Humidity", f"{st.session_state['weather_df']['relative_humidity_2m'].iloc[0]}%")
        w4.metric("Sun Cycle", f"🌅 {st.session_state['sunrise']} | 🌇 {st.session_state['sunset']}")
        
        components.html(f'<iframe width="100%" height="350" src="https://embed.windy.com/embed.html?type=map&location=coordinates&metricWind=km/h&zoom=10&overlay=wind&lat={st.session_state["lat"]}&lon={st.session_state["lon"]}" frameborder="0"></iframe>', height=350)

    with t3:
        st.markdown("### 💰 Smart Grid Financial Arbitrage")
        f1, f2, f3 = st.columns(3)
        f1.metric("Baseline Cost (No Assets)", f"₹{baseline_cost:.2f}")
        f2.metric("Optimized Cost (AI System)", f"₹{optimized_cost:.2f}")
        f3.metric("Estimated Daily Savings", f"₹{savings:.2f}", delta=f"{(savings/baseline_cost*100):.1f}% ROI", delta_color="normal")
        
        st.divider()
        st.markdown("### 🌳 Carbon Neutrality Offset")
        co2_kg = (demand.sum() - sched['Grid'].sum()) * 0.82
        st.metric("Daily CO2 Prevented", f"{co2_kg:.2f} kg", f"Equivalent to {(co2_kg/0.057):.1f} trees daily")
        
        fig_duck = go.Figure()
        fig_duck.add_trace(go.Scatter(x=sched['Hour'], y=demand, name="Gross Demand", line=dict(dash='dot', color='grey')))
        fig_duck.add_trace(go.Scatter(x=sched['Hour'], y=sched['Grid'], name="The Duck Curve", fill='tozeroy', line=dict(color='#E53935', width=4)))
        fig_duck.update_layout(title="Peak Shaving & Load Shifting Visualization", yaxis_title="kWh")
        st.plotly_chart(fig_duck, use_container_width=True)

    with t4:
        if st.button("Generate AI Feature Impact"):
            shap_v = explainer.shap_values(st.session_state['user_feat'].iloc[[0]])
            fig, ax = plt.subplots(figsize=(8,4))
            shap.waterfall_plot(shap.Explanation(values=shap_v[0], base_values=explainer.expected_value, data=st.session_state['user_feat'].iloc[0], feature_names=feature_cols), show=False)
            st.pyplot(fig)
