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
    st.caption("**Top 1% Capstone** | Live 3D Earth & Weather • Virtual Battery • AI Explainability")
with col2:
    st.image("https://cdn-icons-png.flaticon.com/512/2933/2933116.png", width=60)

# 3. SIDEBAR: PROFESSIONAL INPUTS
with st.sidebar:
    st.header("🌍 Location Settings")
    st.caption("Allow GPS access to optimize your local grid:")
    
    # Live Browser Geolocation
    loc = streamlit_geolocation()
    
    lat, lon, resolved_city, country = None, None, "Pune", ""
    
    # Check if user allowed GPS
    if loc and loc.get('latitude') is not None and loc.get('longitude') is not None:
        lat = loc['latitude']
        lon = loc['longitude']
        resolved_city = "Current GPS Location"
        st.success("✅ GPS Locked")
        
        # WOW FEATURE: 3D Interactive Cyber-Globe
        fig_globe = go.Figure(go.Scattergeo(
            lon=[lon], lat=[lat],
            mode='markers+text',
            text=["📍 You Are Here"],
            textposition="bottom center",
            textfont=dict(color="white", size=12, family="Arial Black"),
            marker=dict(size=14, color='#E53935', symbol='circle', line=dict(width=2, color='white'))
        ))
        
       fig_globe.update_geos(
            projection_type="orthographic", # THIS MAKES IT 3D!
            showcoastlines=True, coastlinecolor="#00E5FF",
            showland=True, landcolor="#121212",
            showocean=True, oceancolor="#0A192F",
            showlakes=True, lakecolor="#0A192F",
            showcountries=True, countrycolor="#1E2A38",
            lataxis=dict(showgrid=True, gridcolor="#1E2A38"), # Safely assigned!
            lonaxis=dict(showgrid=True, gridcolor="#1E2A38"), # Safely assigned!
            bgcolor="rgba(0,0,0,0)"
        )
        
        fig_globe.update_layout(
            height=300, 
            margin={"r":0,"t":0,"l":0,"b":0},
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_globe, use_container_width=True, config={'displayModeBar': False})
        
        # Hide the text input since we have GPS
        city_input = st.text_input("City Search (Disabled while GPS active)", value="Using GPS Coordinates", disabled=True)
    else:
        st.info("Or search manually below:")
        city_input = st.text_input("Enter City Name", value="Pune")
    
    st.divider()
    st.header("🏠 Asset Profile")
    num_people = st.slider("Number of residents", 1, 10, 4)
    house_size = st.number_input("House size (m²)", 50, 500, 120, step=10)
    has_ac = st.toggle("❄️ Air Conditioning", value=True)
    
    st.subheader("☀️ Energy Independence")
    has_solar = st.toggle("Rooftop Solar", value=True)
    solar_kw = st.number_input("Solar Array (kW)", 0.0, 20.0, 5.0) if has_solar else 0.0
    
    has_battery = st.toggle("🔋 Home Battery System", value=True)
    battery_capacity = st.number_input("Battery Size (kWh)", 0.0, 30.0, 10.0) if has_battery else 0.0

    st.divider()
    if st.button("🔮 Run AI Optimization", use_container_width=True, type="primary"):
        st.session_state.run_forecast = True
        st.toast("Initializing AI Engine...", icon="🤖")

# 4. HELPER FUNCTIONS
@st.cache_data(ttl=3600)
def get_coordinates(city_name):
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1&format=json"
    try:
        response = requests.get(url).json()
        if "results" in response and len(response["results"]) > 0:
            loc = response["results"][0]
            return loc["latitude"], loc["longitude"], loc["name"], loc.get("country", "")
    except:
        return None, None, None, None
    return None, None, None, None

@st.cache_resource
def load_models():
    try:
        return joblib.load('xgb_model.pkl'), joblib.load('quantile_models.pkl'), joblib.load('shap_explainer_live.pkl'), pickle.load(open('feature_cols.pkl', 'rb'))
    except Exception as e:
        st.error(f"⚠️ Model load error: {e}")
        return None, None, None, None

def engineer_features(df):
    df = df.copy()
    target = 'energy_kwh'
    df['hour'], df['dayofweek'], df['month'] = df.index.hour, df.index.dayofweek, df.index.month
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    for col, period in [('hour', 24), ('dayofweek', 7), ('month', 12)]:
        df[f'{col}_sin'], df[f'{col}_cos'] = np.sin(2 * np.pi * df[col] / period), np.cos(2 * np.pi * df[col] / period)
    df['temp_lag1'] = df['temperature_2m'].shift(1)
    df['hdd'] = np.maximum(18.0 - df['temperature_2m'], 0)
    df['cdd'] = np.maximum(df['temperature_2m'] - 18.0, 0)
    df['cloud_impact'] = df['cloud_cover'] / 100.0
    df['effective_radiation'] = df['shortwave_radiation'] * (1 - df['cloud_impact'])
    
    for lag in [1, 2, 3, 6, 12, 24, 48, 168]: df[f'lag_{lag}h'] = df[target].shift(lag)
    for w in [6, 12, 24, 48]: 
        df[f'roll_mean_{w}h'] = df[target].shift(1).rolling(w).mean()
        df[f'roll_std_{w}h'] = df[target].shift(1).rolling(w).std()
    return df

def generate_smart_schedule(forecast, weather_df, battery_cap):
    hours = [datetime.now().replace(minute=0, second=0) + pd.Timedelta(hours=i) for i in range(24)]
    data = []
    current_battery = 0.0 
    
    for i, (dt, demand) in enumerate(zip(hours, forecast)):
        h = dt.hour
        price = 9 if h < 6 else 15 if h < 9 else 12 if h < 16 else 28 if h < 21 else 11
        
        solar = 0
        if 'shortwave_radiation' in weather_df.columns:
            rad = weather_df['shortwave_radiation'].iloc[i] if i < len(weather_df) else 0
            solar = min(3.5, rad / 1000 * 3.5 * 0.75) if has_solar else 0
            
        net_demand_before_batt = demand - solar
        batt_charge = 0
        batt_discharge = 0
        
        if net_demand_before_batt < 0: 
            batt_charge = min(abs(net_demand_before_batt), battery_cap - current_battery)
            current_battery += batt_charge
            net_grid = 0
        else: 
            if price >= 15: 
                batt_discharge = min(net_demand_before_batt, current_battery)
                current_battery -= batt_discharge
            net_grid = net_demand_before_batt - batt_discharge
            
        rec = "🛑 Peak Avoidance" if price == 28 else "☀️ Charging Battery" if batt_charge > 0 else "🔋 Discharging" if batt_discharge > 0 else "✅ Normal"
        
        data.append({
            'Hour': dt.strftime('%H:%M'), 
            'Demand_kWh': round(demand, 2), 
            'Solar_kWh': round(solar, 2), 
            'Battery_Level': round(current_battery, 2),
            'Grid_Draw_kWh': round(net_grid, 2), 
            'Price_c/kWh': price, 
            'Action': rec
        })
    return pd.DataFrame(data)

# 5. MAIN EXECUTION
xgb_model, quantile_models, explainer, feature_cols = load_models()

if 'run_forecast' in st.session_state and st.session_state.run_forecast:
    
    # Resolve routing: Use GPS if available, else use text search API
    if lat is None or lon is None:
        lat, lon, resolved_city, country = get_coordinates(city_input)
    
    if lat is None:
        st.error("❌ City not found or GPS failed. Please try again.")
    else:
        location_title = f"{resolved_city}, {country}" if country else resolved_city
        with st.spinner(f"🛰️ Syncing live telemetry for {location_title}..."):
            
            cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
            openmeteo = openmeteo_requests.Client(session=retry(cache_session, retries=5))
            response = openmeteo.weather_api("https://api.open-meteo.com/v1/forecast", params={"latitude": lat, "longitude": lon, "hourly": ["temperature_2m", "cloud_cover", "shortwave_radiation", "wind_speed_10m"], "forecast_days": 2, "timezone": "auto"})[0]
            hourly = response.Hourly()
            
            time_index = pd.date_range(start=pd.to_datetime(hourly.Time(), unit="s", utc=True), end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True), freq=pd.Timedelta(seconds=hourly.Interval()), inclusive='left')
            weather_data = {"datetime": time_index}
            for i, var in enumerate(["temperature_2m", "cloud_cover", "shortwave_radiation", "wind_speed_10m"]): weather_data[var] = hourly.Variables(i).ValuesAsNumpy()[:len(time_index)]
            weather_df = pd.DataFrame(weather_data).set_index('datetime').head(24)
            
            dummy = pd.DataFrame(index=weather_df.index).join(weather_df)
            for col in ['Global_active_power','Global_reactive_power','Voltage','Global_intensity','Sub_metering_1','Sub_metering_2','Sub_metering_3','relative_humidity_2m','precipitation']: dummy[col] = 0.0
            dummy['energy_kwh'] = 0.0
            user_feat = engineer_features(dummy)
            for c in feature_cols: 
                if c not in user_feat.columns: user_feat[c] = 0.0
            user_feat = user_feat[feature_cols].astype(np.float32)
            
            scale = (num_people / 4.0) * (house_size / 120.0) * (1.45 if has_ac else 1.0)
            lower, median, upper = quantile_models[0.05].predict(user_feat) * scale, quantile_models[0.50].predict(user_feat) * scale, quantile_models[0.95].predict(user_feat) * scale
            final_pred = np.clip(median, 0, None)
            
            schedule = generate_smart_schedule(final_pred, weather_df, battery_capacity if has_battery else 0)
            
            st.success(f"✅ AI Grid Optimization Complete for **{location_title}**")
            
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Main Forecast", "🌦️ Weather Context", "💰 Savings & Carbon", "🧠 AI Explainability"])
            
            with tab1:
                colA, colB, colC, colD = st.columns(4)
                colA.metric("Raw Demand", f"{final_pred.sum():.1f} kWh")
                colB.metric("Solar Generated", f"{schedule['Solar_kWh'].sum():.1f} kWh")
                colC.metric("Net Grid Draw", f"{schedule['Grid_Draw_kWh'].sum():.1f} kWh", delta=f"-{(final_pred.sum() - schedule['Grid_Draw_kWh'].sum()):.1f} kWh avoided", delta_color="inverse")
                
                max_grid_draw = schedule['Grid_Draw_kWh'].max()
                stress = "🔴 High" if max_grid_draw > 5 else "🟡 Medium" if max_grid_draw > 2.5 else "🟢 Low"
                colD.metric("Grid Stress Level", stress)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=schedule['Hour'], y=final_pred, mode='lines', name='Total Home Demand', line=dict(color='#888', dash='dot')))
                fig.add_trace(go.Scatter(x=schedule['Hour'], y=schedule['Grid_Draw_kWh'], mode='lines+markers', name='Actual Grid Draw', line=dict(color='#E53935', width=4)))
                if has_solar: fig.add_trace(go.Bar(x=schedule['Hour'], y=schedule['Solar_kWh'], name='Solar Yield', marker_color='#FDD835', opacity=0.6))
                if has_battery: fig.add_trace(go.Scatter(x=schedule['Hour'], y=schedule['Battery_Level'], mode='lines', name='Battery SoC', fill='tozeroy', line=dict(color='#4CAF50')))
                
                fig.update_layout(title="24-Hour AI Energy Profile", xaxis_title="Time", yaxis_title="Energy (kWh)", height=500, hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("📅 View Detailed Smart Schedule Data"):
                    st.dataframe(schedule, use_container_width=True, hide_index=True)

            with tab2:
                st.markdown("### 📡 Live Telemetry Data Driving the AI")
                w_col1, w_col2, w_col3 = st.columns(3)
                w_col1.metric("Current Temp", f"{weather_df['temperature_2m'].iloc[0]:.1f}°C")
                w_col2.metric("Cloud Cover", f"{weather_df['cloud_cover'].iloc[0]:.0f}%")
                w_col3.metric("Wind Speed", f"{weather_df['wind_speed_10m'].iloc[0]:.1f} km/h")
                
                fig_w = go.Figure()
                fig_w.add_trace(go.Scatter(x=schedule['Hour'], y=weather_df['temperature_2m'], name="Temp (°C)", line=dict(color="#FF7043")))
                fig_w.add_trace(go.Scatter(x=schedule['Hour'], y=weather_df['shortwave_radiation'], name="Solar Radiation (W/m²)", yaxis="y2", line=dict(color="#FFCA28")))
                fig_w.update_layout(height=400, yaxis2=dict(title="Radiation", overlaying="y", side="right"))
                st.plotly_chart(fig_w, use_container_width=True)

            with tab3:
                cost_no_ai = (final_pred * (schedule['Price_c/kWh'] / 100 * 83)).sum()
                cost_with_ai = (schedule['Grid_Draw_kWh'] * schedule['Price_c/kWh'] / 100 * 83).sum()
                saved = cost_no_ai - cost_with_ai
                co2_saved = (final_pred.sum() - schedule['Grid_Draw_kWh'].sum()) * 0.82
                
                c1, c2 = st.columns(2)
                with c1:
                    st.info("### 💰 Financial ROI")
                    st.metric("Estimated Cost (No Solar/Battery)", f"₹{cost_no_ai:.0f}")
                    st.metric("Cost with AI Smart Grid", f"₹{cost_with_ai:.0f}", delta=f"Saved ₹{saved:.0f}", delta_color="inverse")
                with c2:
                    st.success("### 🌱 Environmental Impact")
                    st.metric("Grid Energy Avoided", f"{(final_pred.sum() - schedule['Grid_Draw_kWh'].sum()):.1f} kWh")
                    st.metric("CO2 Emissions Prevented", f"{co2_saved:.1f} kg 🌳")

            with tab4:
                st.markdown("### 🧠 Real-Time AI Decision Drivers")
                st.caption("This chart shows exactly which weather or time features caused the AI to increase or decrease its energy prediction for the upcoming hours.")
                if st.button("Generate SHAP Waterfall Plot"):
                    with st.spinner("Calculating SHAP values for Hour 0..."):
                        if explainer is not None:
                            shap_values = explainer.shap_values(user_feat.iloc[[0]])
                            fig, ax = plt.subplots(figsize=(8, 5))
                            shap.waterfall_plot(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=user_feat.iloc[0], feature_names=feature_cols), max_display=10, show=False)
                            plt.tight_layout()
                            st.pyplot(fig)
                        else:
                            st.warning("Explainer model missing.")
