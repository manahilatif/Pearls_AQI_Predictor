"""
app.py
------
Streamlit dashboard for the Lahore AQI Predictor.
- Loads model + features from Hopsworks
- Generates 72-hour forecast with 3-day daily summary
- AQI hazard alerts
- EDA tabs
- SHAP explainability
- Model comparison table from Hopsworks
"""

import streamlit as st
import hopsworks
import joblib
import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import requests_cache
from retry_requests import retry
import openmeteo_requests

load_dotenv()

st.set_page_config(page_title="Lahore AQI Predictor", layout="wide", page_icon="🌫️")

# ── FIX #1: Hardcoded FEATURE_COLS — never derived dynamically from df ─────────
# Must match exactly what train_model.py used during training
FEATURE_COLS = [
    "temp", "humidity", "pressure", "clouds", "wind_speed", "wind_deg",
    "pm10", "pm2_5", "co", "no2", "so2", "o3", "nh3",
    "hour", "day", "month", "weekday",
    "aqi_lag_1h", "aqi_lag_24h", "aqi_change_rate",
    "aqi"
]

TARGET_COLS = ["aqi_next_day", "aqi_next_2_days", "aqi_next_3_days"]


# ══════════════════════════════════════════════════════════════════════════════
#  AQI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_aqi_color(aqi_value):
    aqi = float(aqi_value)
    if aqi <= 50:    return "#00E400"
    elif aqi <= 100: return "#FFFF00"
    elif aqi <= 150: return "#FF7E00"
    elif aqi <= 200: return "#FF0000"
    elif aqi <= 300: return "#8F3F97"
    else:            return "#7E0023"


def get_aqi_label(aqi_value):
    aqi = float(aqi_value)
    if aqi <= 50:    return "Good"
    elif aqi <= 100: return "Moderate"
    elif aqi <= 150: return "Unhealthy for Sensitive Groups"
    elif aqi <= 200: return "Unhealthy"
    elif aqi <= 300: return "Very Unhealthy"
    else:            return "HAZARDOUS"


def show_aqi_alert(aqi_value, label="AQI"):
    """Displays a color-coded Streamlit alert based on AQI level."""
    aqi = float(aqi_value)
    msg = f"**{label}: {aqi:.0f}** — {get_aqi_label(aqi)}"
    if aqi <= 50:
        st.success(f"✅ {msg}")
    elif aqi <= 100:
        st.info(f"🟡 {msg}")
    elif aqi <= 150:
        st.warning(f"🟠 {msg} — Sensitive groups should limit outdoor activity.")
    elif aqi <= 200:
        st.warning(f"🔴 {msg} — Everyone may experience health effects.")
    elif aqi <= 300:
        st.error(f"🟣 {msg} — Health alert! Stay indoors.")
    else:
        st.error(f"☠️ {msg} — EMERGENCY. Avoid ALL outdoor activity.")


# ══════════════════════════════════════════════════════════════════════════════
#  WEATHER FORECAST FETCH
# ══════════════════════════════════════════════════════════════════════════════

def get_weather_forecast():
    """Fetches 72-hour hourly weather forecast from Open-Meteo."""
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    client = openmeteo_requests.Client(session=retry_session)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 31.5204,
        "longitude": 74.3587,
        "hourly": [
            "temperature_2m", "relative_humidity_2m", "pressure_msl",
            "cloud_cover", "wind_speed_10m", "wind_direction_10m"
        ],
        "forecast_days": 3,
        "timezone": "auto"
    }

    response = client.weather_api(url, params=params)[0]
    hourly   = response.Hourly()

    dates = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )

    return pd.DataFrame({
        "datetime":   dates,
        "temp":       hourly.Variables(0).ValuesAsNumpy(),
        "humidity":   hourly.Variables(1).ValuesAsNumpy(),
        "pressure":   hourly.Variables(2).ValuesAsNumpy(),
        "clouds":     hourly.Variables(3).ValuesAsNumpy(),
        "wind_speed": hourly.Variables(4).ValuesAsNumpy(),
        "wind_deg":   hourly.Variables(5).ValuesAsNumpy(),
    })


# ══════════════════════════════════════════════════════════════════════════════
#  HOPSWORKS CONNECTION
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_hopsworks_project():
    api_key      = os.getenv("HOPSWORKS_API_KEY")
    project_name = os.getenv("HOPSWORKS_PROJECT_NAME")

    # Streamlit Cloud secrets fallback
    if not api_key:
        api_key = st.secrets.get("HOPSWORKS_API_KEY", None)
    if not project_name:
        project_name = st.secrets.get("HOPSWORKS_PROJECT_NAME", None)

    if not api_key or not project_name:
        return None

    try:
        project = hopsworks.login(project=project_name, api_key_value=api_key)
        return project
    except Exception as e:
        st.error(f"Hopsworks login failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  DATA FETCHING
# ══════════════════════════════════════════════════════════════════════════════

# FIX #2: ttl=3600 so data refreshes hourly matching the feature pipeline schedule
@st.cache_data(ttl=3600)
def get_latest_features(_fs):
    fg    = _fs.get_feature_group(name="aqi_features", version=1)
    query = fg.select_all()
    try:
        df = query.read()
    except Exception:
        df = query.read(read_options={"use_hive": True})

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # Ensure time features exist (safety net)
    if "hour"    not in df.columns: df["hour"]    = df["datetime"].dt.hour
    if "day"     not in df.columns: df["day"]     = df["datetime"].dt.day
    if "month"   not in df.columns: df["month"]   = df["datetime"].dt.month
    if "weekday" not in df.columns: df["weekday"] = df["datetime"].dt.dayofweek

    return df


@st.cache_data(ttl=3600)
def get_model_comparison(_fs):
    """Loads the model comparison table saved by train_model.py."""
    try:
        comp_fg = _fs.get_feature_group(name="model_comparison", version=1)
        return comp_fg.select_all().read()
    except Exception:
        return pd.DataFrame()


def generate_mock_data():
    """Fallback mock data when Hopsworks is unreachable."""
    n = 300
    dates    = pd.date_range(end=datetime.now(), periods=n, freq="h")
    aqi_base = np.random.uniform(80, 250, n)
    return pd.DataFrame({
        "datetime":        dates,
        "temp":            np.random.uniform(10, 40, n),
        "humidity":        np.random.uniform(30, 90, n),
        "pressure":        np.random.uniform(990, 1015, n),
        "wind_speed":      np.random.uniform(0, 8, n),
        "wind_deg":        np.random.uniform(0, 360, n),
        "clouds":          np.random.uniform(0, 100, n),
        "pm10":            np.random.uniform(20, 200, n),
        "pm2_5":           np.random.uniform(10, 150, n),
        "co":              np.random.uniform(0, 3, n),
        "no2":             np.random.uniform(0, 60, n),
        "so2":             np.random.uniform(0, 30, n),
        "o3":              np.random.uniform(0, 80, n),
        "nh3":             np.random.uniform(0, 15, n),
        "hour":            dates.hour,
        "day":             dates.day,
        "month":           dates.month,
        "weekday":         dates.dayofweek,
        "aqi_lag_1h":      np.roll(aqi_base, 1),
        "aqi_lag_24h":     np.roll(aqi_base, 24),
        "aqi_change_rate": np.diff(aqi_base, prepend=aqi_base[0]),
        "aqi":             aqi_base,
    })


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model(_mr, name):
    model_map = {
        "RandomForest": "aqi_randomforest_model",
        "Ridge":        "aqi_ridge_model",
        "XGBoost":      "aqi_xgboost_model",
        "LSTM":         "aqi_lstm_model",
    }

    try:
        # Use version=1 explicitly — never version=None (avoids loading unvalidated models)
        model_meta = _mr.get_model(model_map[name], version=1)
        model_dir  = model_meta.download()
        try:
            metrics = model_meta.metrics or {}
        except Exception:
            metrics = {}
    except Exception as e:
        st.sidebar.error(f"Registry error: {e}")
        return None, {}, None

    # Load scaler if present
    scaler = None
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)

    try:
        if name == "LSTM":
            import tensorflow as tf
            model = tf.keras.models.load_model(
                os.path.join(model_dir, "lstm_model.keras")
            )
        else:
            pkl_files = [f for f in glob.glob(os.path.join(model_dir, "*.pkl"))
                         if "scaler" not in f]
            if not pkl_files:
                st.sidebar.error("No model .pkl found in registry download.")
                return None, metrics, scaler
            model = joblib.load(pkl_files[0])

        return model, metrics, scaler

    except Exception as e:
        st.sidebar.error(f"Model load error: {e}")
        return None, metrics, scaler


# ══════════════════════════════════════════════════════════════════════════════
#  APP LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

st.title("☁️ Lahore AQI Predictor")
st.caption("3-day Air Quality Index forecast powered by Machine Learning | Lahore, Pakistan")

# ── Connect to Hopsworks ───────────────────────────────────────────────────────
project    = get_hopsworks_project()
using_mock = False

if project:
    fs = project.get_feature_store()
    mr = project.get_model_registry()
    with st.spinner("Fetching latest data from Feature Store..."):
        try:
            df = get_latest_features(fs)
        except Exception as e:
            st.warning(f"⚠️ Feature Store read failed — showing mock data. ({e})")
            df         = generate_mock_data()
            using_mock = True
else:
    st.warning("⚠️ Hopsworks unavailable — running in Demo Mode with mock data.")
    st.info("Add `HOPSWORKS_API_KEY` and `HOPSWORKS_PROJECT_NAME` to your `.env` or Streamlit Cloud secrets.")
    df         = generate_mock_data()
    using_mock = True
    mr = None

if using_mock:
    st.info("🔵 **Demo Mode** — predictions use mock data. Deploy to Streamlit Cloud for live data.")

# ── Current conditions banner ──────────────────────────────────────────────────
if not df.empty:
    latest = df.iloc[-1]
    st.markdown("---")
    st.subheader("📍 Current Conditions — Lahore")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🌫️ AQI",      f"{latest['aqi']:.0f}")
    c2.metric("💨 PM2.5",    f"{latest['pm2_5']:.1f} µg/m³")
    c3.metric("🌡️ Temp",     f"{latest['temp']:.1f} °C")
    c4.metric("💧 Humidity", f"{latest['humidity']:.0f}%")
    c5.metric("🕐 Updated",  pd.to_datetime(latest["datetime"]).strftime("%b %d, %H:%M"))

    show_aqi_alert(latest["aqi"], label="Current AQI")

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")
model_name = st.sidebar.selectbox(
    "Select Forecast Model",
    ["RandomForest", "XGBoost", "Ridge", "LSTM"],
    help="RandomForest and XGBoost typically perform best on AQI data."
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Pipeline Status**\n\n"
    "🔄 Feature pipeline: every hour\n\n"
    "🤖 Training pipeline: daily at midnight"
)

# Load model
if mr:
    model, metrics, scaler = load_model(mr, model_name)
else:
    model, metrics, scaler = None, {}, None

if metrics:
    st.sidebar.subheader("📊 Model Metrics")
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            st.sidebar.metric(k, f"{v:.4f}")
else:
    st.sidebar.warning("Model metrics not available.")

# ── Model Comparison Table ─────────────────────────────────────────────────────
if project and not using_mock:
    with st.sidebar.expander("📋 All Models Comparison"):
        comp_df = get_model_comparison(fs)
        if not comp_df.empty:
            st.dataframe(comp_df.set_index("model"), use_container_width=True)
        else:
            st.info("Run training pipeline to populate comparison table.")

# ══════════════════════════════════════════════════════════════════════════════
#  FORECAST SECTION
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.header(f"🔮 72-Hour AQI Forecast  ({model_name})")

if st.button("🚀 Generate Forecast", type="primary"):
    if not model:
        st.error("Model could not be loaded. Check the Model Registry or select a different model.")
    else:
        with st.spinner("Fetching weather forecast and running predictions..."):
            try:
                # 1. Get 72-hour weather forecast from Open-Meteo
                forecast_df = get_weather_forecast()

                # 2. Add time features
                forecast_df["hour"]    = pd.to_datetime(forecast_df["datetime"]).dt.hour
                forecast_df["day"]     = pd.to_datetime(forecast_df["datetime"]).dt.day
                forecast_df["month"]   = pd.to_datetime(forecast_df["datetime"]).dt.month
                forecast_df["weekday"] = pd.to_datetime(forecast_df["datetime"]).dt.dayofweek

                # 3. Persistence assumption: use latest known pollutant values
                latest_rec     = df.iloc[-1]
                pollutant_cols = [
                    "pm10", "pm2_5", "co", "no2", "so2", "o3", "nh3",
                    "aqi", "aqi_lag_1h", "aqi_lag_24h", "aqi_change_rate"
                ]
                for col in pollutant_cols:
                    forecast_df[col] = float(latest_rec.get(col, 0))

                # 4. FIX #1: Use hardcoded FEATURE_COLS — never derived from df
                missing_cols = [c for c in FEATURE_COLS if c not in forecast_df.columns]
                if missing_cols:
                    st.error(f"Missing features: {missing_cols}. Check FEATURE_COLS matches training.")
                    st.stop()

                X_forecast = forecast_df[FEATURE_COLS].copy().fillna(0)

                # 5. Apply scaler for Ridge and LSTM (tree models don't need it)
                if model_name in ["Ridge", "LSTM"] and scaler is not None:
                    X_input = scaler.transform(X_forecast)
                else:
                    X_input = X_forecast.values

                # 6. Predict
                if model_name == "LSTM":
                    import tensorflow as tf
                    # LSTM expects (samples, timesteps, features)
                    X_lstm = X_input.reshape((X_input.shape[0], 1, X_input.shape[1]))
                    preds  = model.predict(X_lstm, verbose=0)
                else:
                    preds = model.predict(X_forecast)

                # 7. Multi-output: use Day 1 prediction as hourly proxy
                if preds.ndim > 1 and preds.shape[1] >= 1:
                    forecast_df["Predicted AQI"] = preds[:, 0].clip(min=0)
                else:
                    forecast_df["Predicted AQI"] = np.array(preds).clip(min=0)

                # FIX #3: Store in session_state for SHAP access later
                st.session_state["X_forecast"]  = X_forecast
                st.session_state["forecast_df"] = forecast_df
                st.session_state["model_name"]  = model_name

                # ── 72-hour line chart ─────────────────────────────────────
                st.subheader("📈 72-Hour Hourly Forecast")
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.plot(forecast_df["datetime"], forecast_df["Predicted AQI"],
                        color="#1A56DB", linewidth=2, label="Predicted AQI")
                ax.axhline(100, color="orange", linestyle="--", alpha=0.5, label="Moderate (100)")
                ax.axhline(200, color="red",    linestyle="--", alpha=0.5, label="Unhealthy (200)")
                ax.fill_between(forecast_df["datetime"],
                                forecast_df["Predicted AQI"], alpha=0.12, color="#1A56DB")
                ax.set_ylabel("AQI")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=30)
                st.pyplot(fig)
                plt.close()

                # ── 3-Day Daily Summary (MISSING FEATURE — now added) ──────
                st.subheader("📅 3-Day Daily AQI Summary")
                forecast_df["date"] = pd.to_datetime(forecast_df["datetime"]).dt.date
                daily = (
                    forecast_df.groupby("date")["Predicted AQI"]
                    .agg(Min="min", Max="max", Mean="mean")
                    .reset_index()
                )

                day_cols = st.columns(min(len(daily), 3))
                for i, (_, row) in enumerate(daily.iterrows()):
                    if i < len(day_cols):
                        color = get_aqi_color(row["Mean"])
                        label = get_aqi_label(row["Mean"])
                        day_cols[i].markdown(
                            f"<div style='background:{color}20;border-left:5px solid {color};"
                            f"padding:14px;border-radius:8px;'>"
                            f"<b>{row['date']}</b><br>"
                            f"<span style='font-size:1.4em;font-weight:bold;color:{color}'>"
                            f"{row['Mean']:.0f}</span> AQI<br>"
                            f"<small>{label}</small><br>"
                            f"<small>Range: {row['Min']:.0f} – {row['Max']:.0f}</small>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

                # ── AQI Alerts ─────────────────────────────────────────────
                st.markdown("### ⚠️ Forecast Alerts")
                peak_aqi = forecast_df["Predicted AQI"].max()
                avg_aqi  = forecast_df["Predicted AQI"].mean()
                show_aqi_alert(peak_aqi, label="Peak Forecast AQI (72h)")
                show_aqi_alert(avg_aqi,  label="Average Forecast AQI (72h)")

                # ── Detailed table ─────────────────────────────────────────
                with st.expander("📋 Full hourly forecast table"):
                    st.dataframe(
                        forecast_df[["datetime", "Predicted AQI", "temp", "humidity", "wind_speed"]]
                        .style.format({
                            "Predicted AQI": "{:.1f}",
                            "temp":          "{:.1f}",
                            "humidity":      "{:.0f}",
                            "wind_speed":    "{:.1f}"
                        }),
                        use_container_width=True
                    )

            except Exception as e:
                st.error(f"Forecast failed: {e}")
                import traceback
                st.text(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
#  EDA SECTION
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.header("📊 Exploratory Data Analysis")

if not df.empty:
    tab1, tab2, tab3, tab4 = st.tabs([
        "Correlations", "Distributions", "Time Trends", "Pollutant Breakdown"
    ])

    with tab1:
        st.subheader("Feature Correlation Heatmap")
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            fig, ax = plt.subplots(figsize=(12, 9))
            sns.heatmap(
                numeric_df.corr(), annot=True, fmt=".1f",
                cmap="coolwarm", center=0, ax=ax,
                annot_kws={"size": 7}
            )
            ax.set_title("Feature Correlation Matrix")
            st.pyplot(fig)
            plt.close()

    with tab2:
        st.subheader("AQI Distribution")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(df["aqi"], kde=True, ax=axes[0], color="#1A56DB")
        axes[0].set_title("AQI Histogram")
        sns.boxplot(y=df["aqi"], ax=axes[1], color="#1A56DB")
        axes[1].set_title("AQI Box Plot")
        st.pyplot(fig)
        plt.close()

    with tab3:
        st.subheader("Historical AQI Trend")
        st.line_chart(df.set_index("datetime")[["aqi", "pm2_5"]])

        if "hour" in df.columns:
            st.subheader("Average AQI by Hour of Day")
            hourly_avg = df.groupby("hour")["aqi"].mean().reset_index()
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.bar(hourly_avg["hour"], hourly_avg["aqi"], color="#1A56DB", alpha=0.8)
            ax.set_xlabel("Hour of Day")
            ax.set_ylabel("Average AQI")
            ax.set_title("Diurnal AQI Pattern")
            st.pyplot(fig)
            plt.close()

        if "month" in df.columns:
            st.subheader("Average AQI by Month")
            monthly_avg = df.groupby("month")["aqi"].mean().reset_index()
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.bar(monthly_avg["month"], monthly_avg["aqi"], color="#E74C3C", alpha=0.8)
            ax.set_xlabel("Month")
            ax.set_ylabel("Average AQI")
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                                 "Jul","Aug","Sep","Oct","Nov","Dec"])
            st.pyplot(fig)
            plt.close()

    with tab4:
        st.subheader("Pollutant Levels Over Time")
        pollutants = ["pm10", "pm2_5", "no2", "o3", "co", "so2"]
        available  = [p for p in pollutants if p in df.columns]
        if available:
            fig, axes = plt.subplots(2, 3, figsize=(14, 7))
            axes = axes.flatten()
            for i, pol in enumerate(available[:6]):
                axes[i].plot(
                    df["datetime"].iloc[-200:],
                    df[pol].iloc[-200:],
                    linewidth=1, color=f"C{i}"
                )
                axes[i].set_title(pol.upper())
                axes[i].tick_params(axis="x", rotation=30)
            for j in range(len(available), 6):
                axes[j].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
if st.checkbox("🔍 Show Model Explainability (SHAP)"):
    st.subheader("SHAP Feature Importance")

    # FIX #3: session_state instead of fragile locals() check
    if "X_forecast" not in st.session_state:
        st.info("Run a forecast first to see SHAP explanations.")
    else:
        X_fc       = st.session_state["X_forecast"]
        saved_name = st.session_state.get("model_name", model_name)

        if model and not X_fc.empty:
            try:
                # Background sample for explainers that need it
                bg_cols    = [c for c in FEATURE_COLS if c in df.columns]
                background = df[bg_cols].dropna().sample(
                    min(50, len(df)), random_state=42
                )

                if saved_name in ["RandomForest", "XGBoost"]:
                    explainer   = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_fc.iloc[:1])
                elif saved_name == "Ridge":
                    explainer   = shap.LinearExplainer(model, background)
                    shap_values = explainer.shap_values(X_fc.iloc[:1])
                else:
                    # LSTM — KernelExplainer (slow but correct)
                    def lstm_predict(X):
                        import tensorflow as tf
                        X_r = X.reshape((X.shape[0], 1, X.shape[1]))
                        return model.predict(X_r, verbose=0)
                    explainer   = shap.KernelExplainer(lstm_predict, background.values[:10])
                    shap_values = explainer.shap_values(X_fc.values[:1])

                if isinstance(shap_values, list):
                    shap_values = shap_values[0]

                st.write("**Feature impact on the next-day AQI prediction:**")
                fig_shap, _ = plt.subplots(figsize=(10, 6))
                shap.summary_plot(
                    shap_values, X_fc.iloc[:1],
                    feature_names=FEATURE_COLS,
                    plot_type="bar", show=False
                )
                st.pyplot(fig_shap)
                plt.close()

            except Exception as e:
                st.warning(f"SHAP could not run: {e}")
                import traceback
                st.text(traceback.format_exc())
        else:
            st.warning("Model not loaded or forecast data unavailable.")