import streamlit as st
import hopsworks
import joblib
import pandas as pd
import numpy as np
import json
import os
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# --- Imports for Weather Forecast ---
import openmeteo_requests
import requests_cache
from retry_requests import retry

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Lahore AQI Predictor", layout="wide", page_icon="🌫️")

# --- Helper Functions ---

def get_weather_forecast():
    """Fetches 72-hour weather forecast from Open-Meteo."""
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 31.5204,
        "longitude": 74.3587,
        "hourly": ["temperature_2m", "relative_humidity_2m", "pressure_msl", "cloud_cover", "wind_speed_10m", "wind_direction_10m"],
        "forecast_days": 3,
        "timezone": "auto"
    }
    
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    
    hourly = response.Hourly()
    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )
    }
    hourly_data["temp"] = hourly.Variables(0).ValuesAsNumpy()
    hourly_data["humidity"] = hourly.Variables(1).ValuesAsNumpy()
    hourly_data["pressure"] = hourly.Variables(2).ValuesAsNumpy()
    hourly_data["clouds"] = hourly.Variables(3).ValuesAsNumpy()
    hourly_data["wind_speed"] = hourly.Variables(4).ValuesAsNumpy()
    hourly_data["wind_deg"] = hourly.Variables(5).ValuesAsNumpy()
    
    forecast_df = pd.DataFrame(data=hourly_data)
    # Align features with model names
    return forecast_df

# --- Connect to Hopsworks ---
@st.cache_resource
def get_hopsworks_project():
    try:
        project = hopsworks.login(
            project=os.getenv("HOPSWORKS_PROJECT_NAME"),
            api_key_value=os.getenv("HOPSWORKS_API_KEY")
        )
        return project
    except Exception as e:
        st.error(f"Could not connect to Hopsworks: {e}")
        return None

project = get_hopsworks_project()

if project:
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    # --- Data Fetching ---
    @st.cache_data
    def get_latest_features():
        aqi_fg = fs.get_feature_group(name="aqi_features", version=1)
        query = aqi_fg.select_all()
        try:
            df = query.read()
        except:
            df = query.read(read_options={"use_hive": True})
            
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime')
        return df

    # --- Mock Data Fallback ---
    def generate_mock_data():
        dates = pd.date_range(end=datetime.now(), periods=200, freq='h')
        return pd.DataFrame({
            'datetime': dates,
            'temp': np.random.uniform(10, 35, 200),
            'humidity': np.random.uniform(30, 90, 200),
            'pressure': np.random.uniform(990, 1010, 200), 
            'wind_speed': np.random.uniform(0, 5, 200),
            'wind_deg': np.random.uniform(0, 360, 200),
            'clouds': np.random.uniform(0, 100, 200),
            'pm10': np.random.uniform(20, 150, 200),
            'pm2_5': np.random.uniform(10, 100, 200),
            'co': np.random.uniform(0, 2, 200),
            'no2': np.random.uniform(0, 50, 200),
            'so2': np.random.uniform(0, 20, 200),
            'o3': np.random.uniform(0, 60, 200),
            'nh3': np.random.uniform(0, 10, 200),
            'aqi': np.random.uniform(50, 200, 200)
        })

    # Fetch Data
    with st.spinner('Fetching data...'):
        try:
            df = get_latest_features()
        except Exception as e:
            st.warning("⚠️ Network issue: Serving Mock Data.")
            df = generate_mock_data()

    # --- UI Header ---
    st.title('☁️ Lahore AQI Predictor')
    
    # Current Status (Top Banner)
    if not df.empty:
        latest = df.iloc[-1]
        st.info(f"📅 **Latest Observation:** {latest['datetime']}")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current AQI", f"{latest['aqi']:.0f}", delta_color="inverse")
        m2.metric("PM2.5", f"{latest['pm2_5']:.1f} µg/m³")
        m3.metric("Temp", f"{latest['temp']:.1f} °C")
        m4.metric("Humidity", f"{latest['humidity']:.0f}%")
    
    # Sidebar: Model and Project Info
    st.sidebar.header("Model Configuration")
    model_name = st.sidebar.selectbox("Select Model", ["RandomForest", "XGBoost", "Ridge", "LSTM"])
    
    st.sidebar.markdown("---") 
    st.sidebar.info("This project fetches live weather logic + pollutant lag features to forecast AQI for the next 72 hours.")

    # --- Model Loading (Robust) ---
    @st.cache_resource
    def load_model(name):
        model_map = {
            "RandomForest": "aqi_randomforest_model",
            "Ridge": "aqi_ridge_model", 
            "XGBoost": "aqi_xgboost_model",
            "LSTM": "aqi_lstm_model"
        }
        
        try:
            # Get Model Object
            model_meta = mr.get_model(model_map[name], version=None)
            model_dir = model_meta.download()
            
            # Safe Metrics Access
            try:
                metrics = model_meta.metrics
                if not metrics: metrics = {}
            except:
                metrics = {} # Default empty if missing attribute
                
        except Exception as e:
            st.sidebar.error(f"Registry Error: {e}")
            return None, {}

        # Load Actual Model File
        try:
            if name == "LSTM":
                import tensorflow as tf
                model = tf.keras.models.load_model(os.path.join(model_dir, "lstm_model.keras"))
            else:
                # Find .pkl file
                import glob
                pkl_files = glob.glob(os.path.join(model_dir, "*.pkl"))
                if pkl_files:
                    model = joblib.load(pkl_files[0])
                else:
                    return None, metrics
            return model, metrics
        except Exception as e:
            st.sidebar.error(f"Load Error: {e}")
            return None, metrics

    model, metrics = load_model(model_name)
    
    if metrics:
        st.sidebar.subheader("Model Metrics")
        st.sidebar.json(metrics)
    else:
        st.sidebar.warning("Model metrics not available.")

    # --- Forecast Loop (72 Hours) ---
    st.header(f"🔮 72-Hour AQI Forecast ({model_name})")
    
    if st.button("Generate Forecast", type="primary"):
        if model:
            with st.spinner("Processing weather forecast & predicting..."):
                try:
                    # 1. Get Weather Forecast
                    forecast_df = get_weather_forecast()
                    
                    # 2. Prepare Features
                    # We need pollutant values. Assumption: Use latest known values (Persistence)
                    latest_rec = df.iloc[-1]
                    pollutants = ['pm10', 'pm2_5', 'co', 'no2', 'so2', 'o3', 'nh3', 'aqi']
                    
                    # Add pollutant columns to forecast_df (constant value)
                    for p in pollutants:
                        if p in latest_rec:
                            forecast_df[p] = latest_rec[p]
                        else:
                            forecast_df[p] = 0 # Safety
                            
                    # Add Time Features
                    forecast_df['datetime'] = pd.to_datetime(forecast_df['date']) # Ensure datetime type
                    forecast_df['day'] = forecast_df['datetime'].dt.day
                    forecast_df['month'] = forecast_df['datetime'].dt.month
                    forecast_df['weekday'] = forecast_df['datetime'].dt.dayofweek
                    
                    # Select columns matching model input (dynamically)
                    # We assume the model was trained on the columns present in 'df' (minus targets)
                    feature_cols = [c for c in df.columns if c not in ['datetime', 'unix_time', 'aqi_next_day', 'aqi_next_2_days', 'aqi_next_3_days']]
                    
                    # Ensure forecast_df has all feature_cols
                    X_forecast = forecast_df[feature_cols].copy()
                    
                    # 3. Predict
                    if model_name == "LSTM":
                        # LSTM Shape: (Samples, 1, Features)
                        X_lstm = X_forecast.values.reshape((X_forecast.shape[0], 1, X_forecast.shape[1]))
                        preds = model.predict(X_lstm)
                    else:
                        preds = model.predict(X_forecast)
                    
                    # Handle Multi-Output: If model returns [Day1, Day2, Day3], take Day1 as the "current hour" prediction proxy
                    if preds.ndim > 1 and preds.shape[1] >= 1:
                        final_preds = preds[:, 0]
                    else:
                        final_preds = preds
                    
                    forecast_df['Predicted AQI'] = final_preds
                    
                    # 4. Visualize
                    st.line_chart(forecast_df.set_index('datetime')['Predicted AQI'])
                    
                    # Display Table
                    st.dataframe(forecast_df[['datetime', 'Predicted AQI', 'temp', 'humidity']].style.format({"Predicted AQI": "{:.1f}", "temp": "{:.1f}", "humidity": "{:.0f}"}))
                    
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    import traceback
                    st.text(traceback.format_exc())

    # --- Full EDA Section ---
    st.markdown("---")
    st.header("📊 Exploratory Data Analysis")
    
    if not df.empty:
        tab1, tab2, tab3 = st.tabs(["Correlations", "Distributions", "Trends"])
        
        with tab1:
            st.subheader("Feature Correlation Heatmap")
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(numeric_df.corr(), annot=False, cmap="coolwarm", ax=ax)
                st.pyplot(fig)
        
        with tab2:
            st.subheader("AQI Distribution")
            fig, ax = plt.subplots()
            sns.histplot(df['aqi'], kde=True, ax=ax, color="purple")
            st.pyplot(fig)
            
        with tab3:
            st.subheader("Time Series Trends")
            st.line_chart(df.set_index('datetime')[['aqi', 'pm2_5']])

    # --- SHAP Explainability ---
    if st.checkbox("Show Model Explainability (SHAP)"):
        st.subheader("🔍 SHAP Feature Importance")
        if model and 'X_forecast' in locals():
            try:
                # Use a small background sample
                background = df[feature_cols].sample(min(50, len(df)))
                
                # Select Explainer
                if model_name in ["RandomForest", "XGBoost"]:
                    explainer = shap.TreeExplainer(model) # Fast
                elif model_name == "Ridge":
                    explainer = shap.LinearExplainer(model, background)
                else: 
                    # Wrapper for generic/LSTM
                    explainer = shap.KernelExplainer(model.predict, background)
                
                # Calculate SHAP for the FIRST hour of forecast
                shap_values = explainer.shap_values(X_forecast.iloc[:1])
                
                # Handle Multi-output SHAP (list of arrays)
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]

                st.write("**Feature Impact on Next Hour Prediction:**")
                fig_shap, ax = plt.subplots()
                shap.summary_plot(shap_values, X_forecast.iloc[:1], plot_type="bar", show=False)
                st.pyplot(fig_shap)
                
            except Exception as e:
                st.warning(f"SHAP could not run: {e}")
        else:
             st.info("Run a forecast first to see SHAP explanations.")
