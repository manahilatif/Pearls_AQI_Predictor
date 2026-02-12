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
    try:
        project = hopsworks.login(
            project=os.getenv("HOPSWORKS_PROJECT_NAME"),
            api_key_value=os.getenv("HOPSWORKS_API_KEY")
        )
        return project
    except Exception as e:
        st.error(f"Failed to connect to Hopsworks: {e}")
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
        # Use simple get_model which fetches the latest version by default or specified version
        # We registered version 1 or 2, get_best_model might be better but get_model is safer for now
        model_meta = mr.get_model(model_name_map[name], version=None) 
        model_dir = model_meta.download()
        
        # Load model object
        if name == "LSTM":
            import tensorflow as tf
            model_path = os.path.join(model_dir, "lstm_model.keras")
            return tf.keras.models.load_model(model_path)
        else:
            model_path = os.path.join(model_dir, f"{name}_model.pkl")
            return joblib.load(model_path)

    if st.sidebar.button("Run Prediction"):
        with st.spinner(f'Loading {model_name} model and generating forecast...'):
            try:
                model = load_model(model_name)
                st.success(f"{model_name} loaded successfully!")
                
                # PREDICTION LOGIC TO BE IMPLEMENTED
                st.info("Prediction logic creates features from the latest data window.")
                
                # Placeholder for prediction
                # In real implementation:
                # 1. Prepare X input from df.tail()
                # 2. model.predict(X)
                # 3. Display result
                
            except Exception as e:
                st.error(f"Error loading model or predicting: {e}")

else:
    st.warning("Please check your .env file for Hopsworks credentials.")
