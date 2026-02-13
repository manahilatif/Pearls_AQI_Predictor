import requests
import pandas as pd
from datetime import datetime

def fetch_weather_data(lat, lon):
    """
    Fetches current weather and AQI data from Open-Meteo API (JSON).
    Matches the schema used in backfill.py.
    """
    # Weather
    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "current": ["temperature_2m", "relative_humidity_2m", "pressure_msl", "wind_speed_10m", "wind_direction_10m", "cloud_cover"]
    }
    
    # AQI
    aqi_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aqi_params = {
        "latitude": lat,
        "longitude": lon,
        "current": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone", "ammonia", "us_aqi"]
    }

    try:
        w_response = requests.get(weather_url, params=weather_params)
        a_response = requests.get(aqi_url, params=aqi_params)
        
        if w_response.status_code != 200 or a_response.status_code != 200:
            print("Error calling Open-Meteo APIs.")
            return {}, {}
            
        return w_response.json(), a_response.json()
        
    except Exception as e:
        print(f"Exception fetching data: {e}")
        return {}, {}

def process_data(weather_data, aqi_data):
    """
    Processes Open-Meteo JSON responses into a structured DataFrame.
    """
    if not weather_data or not aqi_data:
        return pd.DataFrame()
        
    current_w = weather_data.get('current', {})
    current_a = aqi_data.get('current', {})
    
    # Map to schema columns
    data = {
        'datetime': datetime.now(),
        'temp': current_w.get('temperature_2m'),
        'humidity': current_w.get('relative_humidity_2m'),
        'pressure': current_w.get('pressure_msl'),
        'wind_speed': current_w.get('wind_speed_10m'),
        'wind_deg': current_w.get('wind_direction_10m'),
        'clouds': current_w.get('cloud_cover'),
        
        'pm10': current_a.get('pm10'),
        'pm2_5': current_a.get('pm2_5'),
        'co': current_a.get('carbon_monoxide'),
        'no2': current_a.get('nitrogen_dioxide'),
        'so2': current_a.get('sulphur_dioxide'),
        'o3': current_a.get('ozone'),
        'nh3': current_a.get('ammonia'),
        'aqi': current_a.get('us_aqi'), # Standard 0-500 scale
    }
    
    return pd.DataFrame([data])

def engineer_features(df):
    """
    Adds derived features to the DataFrame.
    """
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['unix_time'] = df['datetime'].astype('int64') // 10**6 # Convert nanoseconds to milliseconds
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['weekday'] = df['datetime'].dt.dayofweek
    
    return df
