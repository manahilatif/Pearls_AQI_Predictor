import streamlit as st
import hopsworks
import joblib
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Lahore AQI Predictor", layout="wide")

st.title('☁️ Lahore AQI Predictor')
st.markdown("""
This dashboard predicts the Air Quality Index (AQI) for Lahore using weather and pollutant data.
Select a model from the sidebar to generate forecasts.
""")

# Sidebar for controls
st.sidebar.header('Configuration')

# Connect to Hopsworks
@st.cache_resource
def get_hopsworks_project():
    import time
    for i in range(3):
        try:
            project = hopsworks.login(
                project=os.getenv("HOPSWORKS_PROJECT_NAME"),
                api_key_value=os.getenv("HOPSWORKS_API_KEY")
            )
            return project
        except Exception as e:
            if i < 2:
                time.sleep(2)
                continue
            st.error(f"Failed to connect to Hopsworks after 3 attempts: {e}")
            return None

project = get_hopsworks_project()

if project:
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    # Fetch latest features
    @st.cache_data
    def get_latest_features():
        # Fetch from Feature Store (Online or Offline depending on availability)
        # Online store was disabled for stability, so fetch from Offline
        aqi_fg = fs.get_feature_group(name="aqi_features", version=1)
        query = aqi_fg.select_all()
        
        # Simple read for now
        try:
            df = query.read()
        except:
            df = query.read(read_options={"use_hive": True})
            
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime')
        return df

    with st.spinner('Fetching historical data from Feature Store...'):
        try:
            df = get_latest_features()
            st.success(f"Loaded {len(df)} records.")
            
            # Display Recent Data
            st.subheader("Recent Environmental Data")
            st.dataframe(df.tail(24).sort_values("datetime", ascending=False))
            
            # Plot Historical AQI
            st.subheader("Historical AQI Trend (Last 30 Days)")
            last_30_days = df[df['datetime'] > (df['datetime'].max() - timedelta(days=30))]
            st.line_chart(last_30_days.set_index('datetime')['aqi'])

        except Exception as e:
            st.error(f"Error fetching data: {e}")
            st.stop()

    # Load Models
    model_name = st.sidebar.selectbox("Select Model", ["RandomForest", "Ridge", "XGBoost", "LSTM"])

    @st.cache_resource
    def load_model(name):
        model_name_map = {
            "RandomForest": "aqi_randomforest_model",
            "Ridge": "aqi_ridge_model",
            "XGBoost": "aqi_xgboost_model",
            "LSTM": "aqi_lstm_model"
        }
        
        # Retrieve model from registry
        try:
            # Use simple get_model which fetches the latest version by default or specified version
            # We registered version 1 or 2, get_best_model might be better but get_model is safer for now
            model_meta = mr.get_model(model_name_map[name], version=None) 
            model_dir = model_meta.download()
        except Exception as e:
            st.warning(f"Model {name} not found in registry. Has the training pipeline run successfully?")
            return None
        
        # Load model object
        try:
            if name == "LSTM":
                import tensorflow as tf
                model_path = os.path.join(model_dir, "lstm_model.keras")
                return tf.keras.models.load_model(model_path)
            else:
                model_path = os.path.join(model_dir, f"{name}_model.pkl")
                return joblib.load(model_path)
        except Exception as e:
            st.error(f"Failed to load model file locally: {e}")
            return None

    if st.sidebar.button("Run Prediction"):
        with st.spinner(f'Loading {model_name} model and generating forecast...'):
            try:
                model = load_model(model_name)
                st.success(f"{model_name} loaded successfully!")
                
                # Prepare Input Data
                # The model expects daily averages of features
                # We'll take the average of the last 24 hours of data available
                latest_day = df.tail(24).mean(numeric_only=True)
                
                # Exclude targets and non-features if present in the mean
                # Features used in training:
                # temp, humidity, pressure, wind_speed, wind_deg, clouds, 
                # pm10, pm2_5, co, no2, so2, o3, nh3, aqi, day, month, weekday
                # Note: 'unix_time' was dropped in training.
                
                # Ensure we have all columns in correct order.
                # We rely on the fact that the feature group schema hasn't changed.
                # Ideally we should read the model's input schema, but for now we reconstruct based on training logic.
                
                feature_cols = [c for c in latest_day.index if c not in ['aqi_next_day', 'aqi_next_2_days', 'aqi_next_3_days', 'unix_time', 'datetime']]
                
                # We need to ensure 'day', 'month', 'weekday' are correct for the prediction context (NEXT day?)
                # Actually, the model uses 'current day' features to predict 'next days'.
                # So we use the features of the *latest available day*.
                
                input_data = latest_day[feature_cols].values.reshape(1, -1)
                
                prediction = None
                
                if model_name == "LSTM":
                    # Reshape for LSTM: [samples, time steps, features]
                    # Training used 1 time step
                    input_data = input_data.reshape((1, 1, input_data.shape[1]))
                    prediction = model.predict(input_data)
                else:
                    prediction = model.predict(input_data)
                
                # Display Results
                st.subheader("AQI Forecast")
                cols = st.columns(3)
                
                days = ["Tomorrow", "Day After Tomorrow", "3 Days from Now"]
                
                # Prediction is likely shape (1, 3) or (3,)
                if prediction.ndim > 1:
                    preds = prediction[0]
                else:
                    preds = prediction
                    
                for i, day in enumerate(days):
                    with cols[i]:
                        aqi_val = preds[i]
                        st.metric(label=day, value=f"{aqi_val:.1f}")
                        
                        # Color code based on AQI
                        if aqi_val <= 50:
                            st.success("Good")
                        elif aqi_val <= 100:
                            st.warning("Moderate")
                        elif aqi_val <= 150:
                            st.warning("Unhealthy for Sensitive Groups")
                        else:
                            st.error("Unhealthy/Hazardous")
                            
            except Exception as e:
                st.error(f"Error loading model or predicting: {e}")
                import traceback
                st.text(traceback.format_exc())

else:
    st.warning("Please check your .env file for Hopsworks credentials.")
