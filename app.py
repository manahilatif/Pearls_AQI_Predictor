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
        aqi_fg = fs.get_feature_group(name="aqi_features", version=1)
        query = aqi_fg.select_all()
        
        try:
            print("Attempting to read via Arrow Flight...")
            df = query.read()
        except Exception as e:
            print(f"Arrow Flight failed: {e}. Trying Hive...")
            try:
                df = query.read(read_options={"use_hive": True})
            except Exception as e2:
                print(f"Hive failed: {e2}")
                raise e2
            
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime')
        return df

    def generate_mock_data():
        """Generates mock data if Hopsworks is unreachable for local testing."""
        dates = pd.date_range(end=datetime.now(), periods=100, freq='h')
        mock_df = pd.DataFrame({
            'datetime': dates,
            'temp': np.random.uniform(10, 35, 100),
            'humidity': np.random.uniform(30, 90, 100),
            'pressure': np.random.uniform(990, 1010, 100),
            'wind_speed': np.random.uniform(0, 5, 100),
            'wind_deg': np.random.uniform(0, 360, 100),
            'clouds': np.random.uniform(0, 100, 100),
            'pm10': np.random.uniform(20, 150, 100),
            'pm2_5': np.random.uniform(10, 100, 100),
            'co': np.random.uniform(0, 2, 100),
            'no2': np.random.uniform(0, 50, 100),
            'so2': np.random.uniform(0, 20, 100),
            'o3': np.random.uniform(0, 60, 100),
            'nh3': np.random.uniform(0, 10, 100),
            'aqi': np.random.uniform(50, 200, 100),
            'day': dates.day,
            'month': dates.month,
            'weekday': dates.dayofweek
        })
        return mock_df

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
            st.error(f"Error fetching data from Hopsworks: {e}")
            st.warning("⚠️ Falling back to MOCK DATA for demonstration purposes (Local connection issue).")
            df = generate_mock_data()
            
            # Display Recent Data
            st.subheader("Recent Environmental Data (MOCK)")
            st.dataframe(df.tail(24).sort_values("datetime", ascending=False))
            
            # Plot Historical AQI
            st.subheader("Historical AQI Trend (Last 30 Days) (MOCK)")
            st.line_chart(df.set_index('datetime')['aqi'])

    # Sidebar Information
    st.sidebar.markdown("---")
    st.sidebar.subheader("About Project")
    st.sidebar.markdown(
        """
        **Lahore AQI Predictor** uses advanced Machine Learning to forecast air quality.
        
        **Data Source:** Open-Meteo & OpenWeatherMap (Historical & Live)
        **Target:** PM2.5 & AQI (US EPA Standard)
        """
    )

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
        
        # Retrieve model + metadata from registry
        try:
            model_char = mr.get_model(model_name_map[name], version=None) 
            model_dir = model_char.download()
            metrics = model_char.metrics # Fetch stored metrics
        except Exception as e:
            st.error(f"🔴 Registry connection failed for {name}: {e}")
            st.warning("This is likely a network timeout locally. It will work in the Cloud.")
            return None, None
        
        # Load model object
        try:
            # Debug: Check files
            # st.write(f"Downloaded files: {os.listdir(model_dir)}")
            
            if name == "LSTM":
                import tensorflow as tf
                model_path = os.path.join(model_dir, "lstm_model.keras")
                model = tf.keras.models.load_model(model_path)
            else:
                model_path = os.path.join(model_dir, f"{name}_model.pkl")
                if not os.path.exists(model_path):
                     # Try finding any pkl file
                     files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
                     if files: model_path = os.path.join(model_dir, files[0])
                model = joblib.load(model_path)
            return model, metrics
        except Exception as e:
            st.error(f"🔴 Failed to load model file locally: {e}")
            st.write(f"Contents of {model_dir}: {os.listdir(model_dir)}")
            return None, None

    # Load selected model
    model, metrics = load_model(model_name)

    # Display Metrics in Sidebar
    if metrics:
        st.sidebar.subheader(f"{model_name} Performance")
        st.sidebar.json(metrics)
    else:
        st.sidebar.info("Model metrics not available.")

    # Main Content - EDA
    st.header("📊 Exploratory Data Analysis")
    
    # Feature Correlation Heatmap (using reliable features)
    if not df.empty and 'aqi' in df.columns:
        import seaborn as sns
        
        cols_to_corr = ['temp', 'humidity', 'wind_speed', 'pm2_5', 'pm10', 'no2', 'so2', 'o3', 'aqi']
        avail_cols = [c for c in cols_to_corr if c in df.columns]
        
        if avail_cols:
            st.subheader("Feature Correlations")
            corr = df[avail_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        else:
             st.warning("Not enough columns for correlation analysis.")
    else:
        st.warning("Dataframe is empty or missing AQI column. Cannot show EDA.")

    # Prediction Section
    if st.button("Run Prediction"):
        if model:
            with st.spinner(f'Generated forecast with {model_name}...'):
                try:
                    # Input Data Prep (Same as before)
                    latest_day = df.tail(24).mean(numeric_only=True)
                    feature_cols = [c for c in latest_day.index if c not in ['aqi_next_day', 'aqi_next_2_days', 'aqi_next_3_days', 'unix_time', 'datetime']]
                    input_data = latest_day[feature_cols].values.reshape(1, -1)
                    
                    # Store for SHAP (DataFrame format for sklearn)
                    input_df = pd.DataFrame([latest_day[feature_cols]])

                    prediction = None
                    if model_name == "LSTM":
                        input_data_lstm = input_data.reshape((1, 1, input_data.shape[1]))
                        prediction = model.predict(input_data_lstm)
                    else:
                        prediction = model.predict(input_data)
                    
                    # Display Results
                    st.subheader("🔮 3-Day AQI Forecast")
                    cols = st.columns(3)
                    
                    days = ["Tomorrow", "Day After Tomorrow", "3 Days from Now"]
                    preds = prediction[0] if prediction.ndim > 1 else prediction
                        
                    for i, day in enumerate(days):
                        with cols[i]:
                            aqi_val = preds[i]
                            st.metric(label=day, value=f"{aqi_val:.1f}")
                            if aqi_val <= 50: st.success("Good")
                            elif aqi_val <= 100: st.warning("Moderate")
                            elif aqi_val <= 150: st.warning("Unhealthy for SG")
                            else: st.error("Unhealthy")

                    # Visualizing History + Forecast
                    st.subheader("📉 History + 🚀 Future Projection")
                    
                    # Prepare data for plot
                    future_dates = [datetime.now() + timedelta(days=i+1) for i in range(3)]
                    history_df = df.tail(7) # Last 7 days
                    future_df = pd.DataFrame({
                        'datetime': future_dates,
                        'aqi': preds,
                        'type': ['Forecast'] * 3
                    })
                    
                    # Combine for plotting (simplified)
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(history_df['datetime'], history_df['aqi'], label='History (Actual)', marker='o')
                    ax.plot(future_df['datetime'], future_df['aqi'], label='Forecast (Predicted)', linestyle='--', marker='x', color='red')
                    ax.set_title("AQI Trend: Past 7 Days -> Next 3 Days")
                    ax.legend()
                    st.pyplot(fig)
                    
                    st.info("ℹ️ **Note:** The model uses *past* data (History) to predict *future* AQI. The chart above shows how the trend is expected to continue.")
                    
                    # SHAP Analysis (Beta)
                    if model_name in ["RandomForest", "XGBoost", "Ridge"]:
                        st.subheader("🔍 Model Explainability (SHAP)")
                        import shap
                        
                        # Use KernelExplainer as generic fallback or TreeExplainer if possible
                        # Ideally providing background data (from df) is better
                        # For speed, we use a small background sample
                        background = df[feature_cols].sample(min(10, len(df)))
                        
                        if model_name == "Ridge":
                            explainer = shap.LinearExplainer(model, background)
                        else:
                             # Tree explainer might require specific object, use Kernel for robust app usage
                            explainer = shap.KernelExplainer(model.predict, background)
                            
                        shap_values = explainer.shap_values(input_df)
                        
                        # Visualize first output (Tomorrow) for multi-output
                        # Check shape of shap_values
                        # It might be a list of arrays (one for each target)
                        if isinstance(shap_values, list):
                             sv = shap_values[0] # Tomorrow
                             st.write("**Feature Impact on Tomorrow's Prediction:**")
                        else:
                             sv = shap_values
                        
                        # Summary Plot or Force Plot
                        # Force plot needs JS, might be tricky in Streamlit without component
                        # Let's use matplotlib plot
                        fig_shap, ax_shap = plt.subplots()
                        shap.summary_plot(sv, input_df, plot_type="bar", show=False)
                        st.pyplot(fig_shap)

                except Exception as e:
                    st.error(f"Prediction Error: {e}")
                    # import traceback
                    # st.text(traceback.format_exc())
        else:
             st.error("Model not loaded.")

else:
    st.warning("Please check your .env file for Hopsworks credentials.")
