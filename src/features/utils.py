import requests
import pandas as pd
from datetime import datetime


def fetch_weather_data(lat, lon):
    """
    Fetches hourly weather + AQI data from Open-Meteo (past 2 days + today).
    Uses hourly (not current) so we have enough rows to compute lag features.
    """
    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "temperature_2m", "relative_humidity_2m", "pressure_msl",
            "cloud_cover", "wind_speed_10m", "wind_direction_10m"
        ],
        "past_days": 2,
        "forecast_days": 1,
        "timezone": "auto"
    }

    aqi_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aqi_params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
            "sulphur_dioxide", "ozone", "ammonia", "us_aqi"
        ],
        "past_days": 2,
        "forecast_days": 1,
        "timezone": "auto"
    }

    try:
        w_response = requests.get(weather_url, params=weather_params, timeout=10)
        a_response = requests.get(aqi_url, params=aqi_params, timeout=10)

        if w_response.status_code != 200 or a_response.status_code != 200:
            print(f"API error — weather: {w_response.status_code}, aqi: {a_response.status_code}")
            return {}, {}

        return w_response.json(), a_response.json()

    except Exception as e:
        print(f"Exception fetching data: {e}")
        return {}, {}


def process_data(weather_data, aqi_data):
    """
    Converts Open-Meteo hourly JSON responses into a merged DataFrame.
    """
    if not weather_data or not aqi_data:
        return pd.DataFrame()

    # Parse weather hourly
    w_hourly = weather_data.get("hourly", {})
    weather_df = pd.DataFrame({
        "datetime":   w_hourly.get("time", []),
        "temp":       w_hourly.get("temperature_2m", []),
        "humidity":   w_hourly.get("relative_humidity_2m", []),
        "pressure":   w_hourly.get("pressure_msl", []),
        "clouds":     w_hourly.get("cloud_cover", []),
        "wind_speed": w_hourly.get("wind_speed_10m", []),
        "wind_deg":   w_hourly.get("wind_direction_10m", []),
    })

    # Parse AQI hourly
    a_hourly = aqi_data.get("hourly", {})
    aqi_df = pd.DataFrame({
        "datetime": a_hourly.get("time", []),
        "pm10":     a_hourly.get("pm10", []),
        "pm2_5":    a_hourly.get("pm2_5", []),
        "co":       a_hourly.get("carbon_monoxide", []),
        "no2":      a_hourly.get("nitrogen_dioxide", []),
        "so2":      a_hourly.get("sulphur_dioxide", []),
        "o3":       a_hourly.get("ozone", []),
        "nh3":      a_hourly.get("ammonia", []),
        "aqi":      a_hourly.get("us_aqi", []),
    })

    # Merge on datetime and drop rows with missing AQI
    merged = pd.merge(weather_df, aqi_df, on="datetime", how="inner")
    merged["datetime"] = pd.to_datetime(merged["datetime"])
    merged = merged.dropna(subset=["aqi"]).reset_index(drop=True)

    return merged


def engineer_features(df):
    """
    Adds time-based and lag features required by the model.
    """
    if df.empty:
        return df

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # FIX: unix_time in seconds (not milliseconds)
    df["unix_time"] = df["datetime"].astype("int64") // 10**9

    # Time features
    df["hour"]    = df["datetime"].dt.hour
    df["day"]     = df["datetime"].dt.day
    df["month"]   = df["datetime"].dt.month
    df["weekday"] = df["datetime"].dt.dayofweek

    # FIX: Lag features (were missing entirely)
    df["aqi_lag_1h"]      = df["aqi"].shift(1)
    df["aqi_lag_24h"]     = df["aqi"].shift(24)
    df["aqi_change_rate"] = df["aqi"].diff()

    # Drop rows with NaN from shifting
    df = df.dropna().reset_index(drop=True)

    return df