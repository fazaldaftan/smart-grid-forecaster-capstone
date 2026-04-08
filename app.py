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

# 2. HELPER FUNCTIONS
@st.cache_data(ttl=3600)
def get_coordinates(city_name):
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1&format=json"
    try:
        response = requests.get(url).json()
        if "results" in response and len(response["results"]) > 0:
            loc = response["results"][0]
            return loc["latitude"], loc["longitude"], loc["name"], loc.get("country", "")
    except: pass
    return None, None, None, None

@st.cache_resource
def load_models():
    try: 
        xgb = joblib.load('xgb_model.pkl')
        quantiles = joblib.load('quantile_models.pkl')
        explainer = joblib.load('shap_explainer_live.pkl')
        with open('feature_cols.pkl', 'rb') as f:
            feature_cols = pickle.load(f)
        return xgb, quantiles, explainer, feature_cols
    except Exception as e:
        st.error(f"⚠️ CRITICAL MODEL ERROR: {e}")
        st.info("Check if all .pkl files are pushed to GitHub, and ensure they aren't corrupted by Git LFS.")
        return None, None, None, None

def engineer_features(df, feature_cols):
    df = df.copy()
    df['hour'], df['dayofweek'], df['month'] = df.index.hour, df.index.dayofweek, df.index.month
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    for col, period in [('hour', 24), ('dayofweek', 7), ('month', 12)]:
        df[f'{col}_sin'], df[f'{col}_cos'] = np.sin(2 * np.pi * df[col] / period), np.cos(2 * np.pi * df[col] / period)
    
    df['temp_lag1'] = df['temperature_2m'].shift(1).bfill()
    df['hdd'] = np.maximum(18.0 - df['temperature_2m'], 0)
    df['cdd'] = np.maximum(df['temperature_2m'] - 18.0, 0)
    df['cloud_impact'] = df['cloud_cover'] / 100.0
    df['effective_radiation'] = df['shortwave_radiation'] * (1 - df['cloud_impact'])
    
    for lag in [1, 2, 3, 6, 12, 24, 48, 168]: df[f'lag_{lag}h'] = 0.0
    for w in [6, 12, 24, 48]: 
        df[f'roll_mean_{w}h'] = 0.0
        df[f'roll_std_{w}h'] = 0.0
        
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
            
    return df[feature_cols].astype(np.float32)

def generate_smart_schedule(forecast, weather_df, current_solar_kw, current_batt_cap):
    hours = [datetime.now().replace(minute=0, second=0) + pd.Timedelta(hours=i) for i in range(24)]
    data = []
    current_battery = 0.0 
    
    for i, (dt, demand) in enumerate(zip(hours, forecast)):
        h = dt.hour
        price = 7.47 if h < 6 else 12.45 if h < 9 else 9.96 if h < 16 else 23.24 if h < 21 else 9.13
        
        solar = 0
        if 'shortwave_radiation' in weather_df.columns:
            rad = weather_df['shortwave_radiation'].iloc[i] if i < len(weather_df) else 0
            solar = min(current_solar_kw, rad / 1000 * current_solar_kw * 0.75) if current_solar_kw > 0 else 0
            
        net_demand_before_batt = demand - solar
        batt_charge, batt_discharge = 0, 0
        
        if net_demand_before_batt < 0: 
            batt_charge = min(abs(net_demand_before_batt), current_batt_cap - current_battery)
            current_battery += batt_charge
            net_grid = 0
        else: 
            if price >= 15: 
                batt_discharge = min(net_demand_before_batt, current_battery)
                current_battery -= batt_discharge
            net_grid = net_demand_before_batt - batt_discharge
            
        rec = "🛑 Peak Avoidance" if price == 23.24 else "☀️ Charging Battery" if batt_charge > 0 else "🔋 Discharging" if batt_discharge > 0 else "✅ Normal"
        data.append({'Hour': dt.strftime('%H:%M'), 'Demand_kWh': round(demand, 2), 'Solar_kWh': round(solar, 2), 'Battery_Level': round(current_battery, 2), 'Grid_Draw_kWh': round(net_grid, 2), 'Price_c/kWh': price, 'Action': rec})
    return pd.DataFrame(data)

# 3. HEADER
col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.title("⚡ AI Smart Grid Forecaster & Optimizer")
    st.caption("**Top 1% Capstone** | Live 3D Earth • Live AQI • Financial Arbitrage • AI Duck Curve")
with col2:
    st.image("https://cdn-icons-png.flaticon.com/512/2933/2933116.png", width=60)

# 4. SIDEBAR: PROFESSIONAL INPUTS
with st.sidebar:
    st.header("🌍 Location Settings")
    search_mode = st.radio("Choose Location Method:", ["Search by City", "Use Live GPS"])
    
    lat, lon, resolved_city, country = None, None, "", ""
    
    if search_mode == "Use Live GPS":
        loc = streamlit_geolocation()
        if loc and loc.get('latitude') is not None and loc.get('longitude') is not None:
            lat, lon = loc['latitude'], loc['longitude']
            resolved_city = "Current GPS Location"
            st.success("✅ GPS Locked")
        else:
            st.info("Awaiting GPS Permission...")
    else:
        city_input = st.text_input("Enter City Name", value="Pune")
        lat, lon, resolved_city, country = get_coordinates(city_input)
        if lat:
            st.success(f"✅ Found: {resolved_city}, {country}")
        else:
            st.error("❌ City not found.")

    if lat is not None and lon is not None:
        fig_globe = go.Figure(go.Scattergeo(
            lon=[lon], lat=[lat], mode='markers+text', text=["📍 Target Area"], textposition="bottom center",
            textfont=dict(color="white", size=12, family="Arial Black"),
            marker=dict(size=14, color='#E53935', symbol='circle', line=dict(width=2, color='white'))
        ))
        fig_globe.update_geos(
            projection_type="orthographic", showcoastlines=True, coastlinecolor="#00E5FF",
            showland=True, landcolor="#121212", showocean=True, oceancolor="#0A192F",
            showlakes=True, lakecolor="#0A192F", showcountries=True, countrycolor="#1E2A38",
            lataxis=dict(showgrid=True, gridcolor="#1E2A38"), lonaxis=dict(showgrid=True, gridcolor="#1E2A38")
        )
        fig_globe.update_layout(height=300, margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_globe, use_container_width=True, config={'displayModeBar': False})
    
    st.divider()
    st.header("🏠 Live Asset Profile")
    st.caption("Moving these sliders updates the dashboard instantly!")
    num_people = st.slider("Number of residents", 1, 10, 4)
    house_size = st.number_input("House size (m²)", 50, 500, 120, step=10)
    has_ac = st.toggle("❄️ Air Conditioning", value=True)
    
    st.subheader("☀️ Energy Independence")
    has_solar = st.toggle("Rooftop Solar", value=True)
    solar_kw = st.slider("Solar Array (kW)", 0.0, 20.0, 5.0) if has_solar else 0.0
    
    has_battery = st.toggle("🔋 Home Battery System", value=True)
    battery_capacity = st.slider("Battery Size (kWh)", 0.0, 30.0, 10.0) if has_battery else 0.0

    st.divider()
    if st.button("🔮 Run AI Optimization", use_container_width=True, type="primary"):
        if lat is not None:
            st.session_state.fetch_data = True
            st.toast("Fetching live telemetry and running AI...", icon="🤖")
        else:
            st.error("Please resolve the location first.")

# 5. MAIN EXECUTION
xgb_model, quantile_models, explainer, feature_cols = load_models()

# FETCH DATA ONLY ONCE
if st.session_state.get('fetch_data', False):
    if feature_cols is None or quantile_models is None:
        st.error("❌ Cannot run AI Optimization: Models failed to load. Please check the CRITICAL ERROR message above.")
        st.session_state.fetch_data = False
    else:
        location_title = f"{resolved_city}, {country}" if country else resolved_city
        st.session_state['location_title'] = location_title
        st.session_state['lat'] = lat
        st.session_state['lon'] = lon
        
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        openmeteo = openmeteo_requests.Client(session=retry(cache_session, retries=5))
        params = {
            "latitude": lat, "longitude": lon, 
            "hourly": ["temperature_2m", "cloud_cover", "shortwave_radiation", "wind_speed_10m", "wind_gusts_10m", "relative_humidity_2m"], 
            "forecast_days": 2, "timezone": "auto"
        }
        response = openmeteo.weather_api("https://api.open-meteo.com/v1/forecast", params=params)[0]
        hourly = response.Hourly()
        
        time_index = pd.date_range(start=pd.to_datetime(hourly.Time(), unit="s", utc=True), end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True), freq=pd.Timedelta(seconds=hourly.Interval()), inclusive='left')
        weather_data = {"datetime": time_index}
        for i, var in enumerate(["temperature_2m", "cloud_cover", "shortwave_radiation", "wind_speed_10m", "wind_gusts_10m", "relative_humidity_2m"]): 
            weather_data[var] = hourly.Variables(i).ValuesAsNumpy()[:len(time_index)]
        weather_df = pd.DataFrame(weather_data).set_index('datetime').head(24)
        
        try:
            daily_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,temperature_2m_min,sunrise,sunset&timezone=auto"
            daily_data = requests.get(daily_url).json()
            st.session_state['today_high'] = daily_data['daily']['temperature_2m_max'][0]
            st.session_state['today_low'] = daily_data['daily']['temperature_2m_min'][0]
            st.session_state['sunrise'] = datetime.fromisoformat(daily_data['daily']['sunrise'][0]).strftime('%I:%M %p')
            st.session_state['sunset'] = datetime.fromisoformat(daily_data['daily']['sunset'][0]).strftime('%I:%M %p')
        except:
            st.session_state['today_high'], st.session_state['today_low'], st.session_state['sunrise'], st.session_state['sunset'] = 0, 0, "N/A", "N/A"

        try:
            aqi_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&current=us_aqi,pm10,pm2_5&timezone=auto"
            aqi_data = requests.get(aqi_url).json()
            st.session_state['aqi_val'] = aqi_data['current']['us_aqi']
            st.session_state['pm10'] = aqi_data['current']['pm10']
            st.session_state['pm25'] = aqi_data['current']['pm2_5']
        except:
            st.session_state['aqi_val'], st.session_state['pm10'], st.session_state['pm25'] = 0, 0, 0

        dummy = pd.DataFrame(index=weather_df.index).join(weather_df)
        for col in ['Global_active_power','Global_reactive_power','Voltage','Global_intensity','Sub_metering_1','Sub_metering_2','Sub_metering_3','precipitation']: dummy[col] = 0.0
        dummy['energy_kwh'] = 0.0
        
        user_feat = engineer_features(dummy, feature_cols)
        
        st.session_state['weather_df'] = weather_df
        st.session_state['base_median'] = quantile_models[0.50].predict(user_feat)
        st.session_state['user_feat'] = user_feat
        st.session_state['app_ready'] = True
        st.session_state['fetch_data'] = False

# REACTIVE UI (With Bulletproof Session State Check)
if st.session_state.get('app_ready', False):
    if 'base_median' not in st.session_state or 'weather_df' not in st.session_state:
        st.warning("⚠️ Session memory was cleared. Please click 'Run AI Optimization' in the sidebar to refresh the dashboard.")
    else:
        scale = (num_people / 4.0) * (house_size / 120.0) * (1.45 if has_ac else 1.0)
        final_pred = np.clip(st.session_state['base_median'] * scale, 0, None)
        
        active_solar = solar_kw if has_solar else 0
        active_battery = battery_capacity if has_battery else 0
        weather_df = st.session_state['weather_df']
        schedule = generate_smart_schedule(final_pred, weather_df, active_solar, active_battery)
        
        st.success(f"✅ AI Grid Optimization Active for **{st.session_state['location_title']}**")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Main Forecast", "📡 Advanced Weather", "💰 ROI & Carbon", "🧠 AI Explainability"])
        
        with tab1:
            colA, colB, colC, colD = st.columns(4)
            colA.metric("Raw Demand", f"{final_pred.sum():.1f} kWh")
            colB.metric("Solar Generated", f"{schedule['Solar_kWh'].sum():.1f} kWh")
            colC.metric("Net Grid Draw", f"{schedule['Grid_Draw_kWh'].sum():.1f} kWh", delta=f"-{(final_pred.sum() - schedule['Grid_Draw_kWh'].sum()):.1f} kWh avoided", delta_color="inverse")
            max_grid_draw = schedule['Grid_Draw_kWh'].max()
            colD.metric("Grid Stress Level", "🔴 High" if max_grid_draw > 5 else "🟡 Medium" if max_grid_draw > 2.5 else "🟢 Low")

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
            st.markdown("### 🛰️ Live Telemetry & Environmental Data")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Current Temp", f"{weather_df['temperature_2m'].iloc[0]:.1f}°C")
            c2.metric("Today's Range", f"H: {st.session_state['today_high']:.1f}° | L: {st.session_state['today_low']:.1f}°")
            c3.metric("Relative Humidity", f"{weather_df['relative_humidity_2m'].iloc[0]:.0f}%")
            c4.metric("Sunrise / Sunset", f"🌅 {st.session_state['sunrise']} | 🌇 {st.session_state['sunset']}")

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Sustained Wind", f"{weather_df['wind_speed_10m'].iloc[0]:.1f} km/h", f"Gusts up to {weather_df['wind_gusts_10m'].iloc[0]:.1f} km/h", delta_color="off")
            
            aqi_val = st.session_state['aqi_val']
            aqi_status = "🟢 Good" if aqi_val <= 50 else "🟡 Moderate" if aqi_val <= 100 else "🟠 Unhealthy" if aqi_val <= 150 else "🔴 Hazardous"
            c6.metric("Air Quality Index (AQI)", f"{aqi_val}", aqi_status, delta_color="off")
            c7.metric("Particulate Matter (µg/m³)", f"PM2.5: {st.session_state['pm25']}", f"PM10: {st.session_state['pm10']}", delta_color="off")
            c8.metric("Cloud Cover", f"{weather_df['cloud_cover'].iloc[0]:.0f}%")

            st.divider()

            col_chart, col_map = st.columns([1.2, 1])
            with col_chart:
                st.markdown("#### 🌡️ 24-Hour Temp & Radiation")
                fig_w = go.Figure()
                fig_w.add_trace(go.Scatter(x=schedule['Hour'], y=weather_df['temperature_2m'], name="Temp (°C)", line=dict(color="#FF7043")))
                fig_w.add_trace(go.Scatter(x=schedule['Hour'], y=weather_df['shortwave_radiation'], name="Solar Radiation (W/m²)", yaxis="y2", line=dict(color="#FFCA28")))
                fig_w.update_layout(height=400, yaxis2=dict(title="Radiation (W/m²)", overlaying="y", side="right"), margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_w, use_container_width=True)
                
            with col_map:
                st.markdown("#### 🌬️ Live Interactive Wind Map")
                windy_html = f"""
                <iframe width="100%" height="400" 
                src="https://embed.windy.com/embed.html?type=map&location=coordinates&metricRain=mm&metricTemp=°C&metricWind=km/h&zoom=10&overlay=wind&product=ecmwf&level=surface&lat={st.session_state['lat']}&lon={st.session_state['lon']}" 
                frameborder="0" style="border-radius: 10px;"></iframe>
                """
                components.html(windy_html, height=400)

        with tab3:
            baseline_cost = (schedule['Demand_kWh'] * schedule['Price_c/kWh']).sum()
            optimized_cost = (schedule['Grid_Draw_kWh'] * schedule['Price_c/kWh']).sum()
            savings = baseline_cost - optimized_cost
            roi_percentage = (savings / baseline_cost * 100) if baseline_cost > 0 else 0
            
            daily_offset = final_pred.sum() - schedule['Grid_Draw_kWh'].sum()
            co2_saved_kg = daily_offset * 0.82
            trees_equivalent = co2_saved_kg / 0.057
            
            st.markdown("### 💰 Smart Grid Financial Arbitrage")
            st.caption("The AI calculates dynamic Time-of-Use pricing. Notice how the battery saves money by avoiding expensive evening grid rates.")
            f1, f2, f3 = st.columns(3)
            f1.metric("Baseline Cost (No Assets)", f"₹{baseline_cost:.2f}")
            f2.metric("Optimized Cost (AI System)", f"₹{optimized_cost:.2f}")
            f3.metric("Estimated Daily Savings", f"₹{savings:.2f}", delta=f"{roi_percentage:.1f}% ROI", delta_color="normal")
            
            st.divider()
            
            st.markdown("### 🌳 Carbon Neutrality Offset (Duck Curve)")
            c1, c2, c3 = st.columns(3)
            c1.info(f"### 📉 Daily Offset\n**{daily_offset:.1f} kWh** avoided")
            c2.success(f"### 🏭 Carbon Prevented\n**{co2_saved_kg:.1f} kg** of CO2")
            c3.warning(f"### 🌳 Trees Equivalent\n**{trees_equivalent:.2f} units** of absorption")

            fig_duck = go.Figure()
            fig_duck.add_trace(go.Scatter(x=schedule['Hour'], y=final_pred, mode='lines', name='Gross Baseline Demand', line=dict(color='rgba(100,100,100,0.5)', width=3, dash='dash')))
            fig_duck.add_trace(go.Scatter(x=schedule['Hour'], y=schedule['Grid_Draw_kWh'], mode='lines', name='Optimized Grid Draw', line=dict(color='#E53935', width=4), fill='tozeroy', fillcolor='rgba(229, 57, 53, 0.1)'))
            fig_duck.update_layout(height=400, hovermode="x unified", xaxis_title="Time of Day", yaxis_title="Grid Dependency (kWh)", margin=dict(t=10))
            st.plotly_chart(fig_duck, use_container_width=True)

        with tab4:
            st.markdown("### 🧠 AI Feature Impact")
            if st.button("Generate SHAP Waterfall"):
                with st.spinner("Analyzing Hour 0..."):
                    if explainer is not None:
                        shap_values = explainer.shap_values(st.session_state['user_feat'].iloc[[0]])
                        fig, ax = plt.subplots(figsize=(8, 5))
                        shap.waterfall_plot(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=st.session_state['user_feat'].iloc[0], feature_names=feature_cols), max_display=10, show=False)
                        plt.tight_layout()
                        st.pyplot(fig)
