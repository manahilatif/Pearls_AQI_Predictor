import requests
import pandas as pd
from datetime import datetime

def fetch_weather_data(lat, lon, api_key):
    """
    Fetches current weather and AQI data from OpenWeatherMap API.
    """
    # Weather
    weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    weather_response = requests.get(weather_url)
    if weather_response.status_code != 200:
        print(f"Error fetching weather data: {weather_response.status_code} - {weather_response.text}")
        weather_data = {}
    else:
        weather_data = weather_response.json()

    # AQI
    aqi_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
    aqi_response = requests.get(aqi_url)
    if aqi_response.status_code != 200:
        print(f"Error fetching AQI data: {aqi_response.status_code} - {aqi_response.text}")
        aqi_data = {}
    else:
        aqi_data = aqi_response.json()

    return weather_data, aqi_data

def process_data(weather_data, aqi_data):
    """
    Processes raw API responses into a structured DataFrame.
    """
    # Extract Weather features
    main = weather_data.get('main', {})
    wind = weather_data.get('wind', {})
    clouds = weather_data.get('clouds', {})
    
    # Extract AQI features
    aqi_list = aqi_data.get('list', [{}])[0]
    components = aqi_list.get('components', {})
    main_aqi = aqi_list.get('main', {})

    data = {
        'datetime': datetime.now(),
        'temp': main.get('temp'),
        'pressure': main.get('pressure'),
        'humidity': main.get('humidity'),
        'wind_speed': wind.get('speed'),
        'wind_deg': wind.get('deg'),
        'clouds': clouds.get('all'),
        'aqi': main_aqi.get('aqi'),
        'co': components.get('co'),
        'no': components.get('no'),
        'no2': components.get('no2'),
        'o3': components.get('o3'),
        'so2': components.get('so2'),
        'pm2_5': components.get('pm2_5'),
        'pm10': components.get('pm10'),
        'nh3': components.get('nh3'),
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
