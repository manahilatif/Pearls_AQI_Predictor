import sys
import os
# Add 'src' to sys.path to allow imports from features module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import hopsworks
from dotenv import load_dotenv
from datetime import datetime, timedelta
from src.features.utils import engineer_features

# Load environment variables
load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")

LAT = 31.5204 # Lahore
LON = 74.3587

def fetch_historical_data(start_date, end_date):
    """
    Fetches historical weather and AQI data from Open-Meteo
    """
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # Weather API
    weather_url = "https://archive-api.open-meteo.com/v1/archive"
    weather_params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["temperature_2m", "relative_humidity_2m", "pressure_msl", "surface_pressure", "wind_speed_10m", "wind_direction_10m", "cloud_cover"]
    }
    
    # AQI API (Air Quality)
    aqi_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aqi_params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone", "ammonia", "us_aqi"]
    }

    print(f"Fetching historical data from {start_date} to {end_date}...")
    
    try:
        weather_responses = openmeteo.weather_api(weather_url, params=weather_params)
        aqi_responses = openmeteo.weather_api(aqi_url, params=aqi_params)
    except Exception as e:
        print(f"Error calling Open-Meteo API: {e}")
        return pd.DataFrame() # Return empty on error to avoid crash

    # Process Weather Data
    if not weather_responses:
         print("No weather data returned.")
         return pd.DataFrame()

    response = weather_responses[0]
    hourly = response.Hourly()
    print(f"Weather data points: {hourly.Variables(0).ValuesAsNumpy().size}")
    
    hourly_data = {"datetime": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}
    
    weather_data = {
        "datetime": hourly_data["datetime"],
        "temp": hourly.Variables(0).ValuesAsNumpy(),
        "humidity": hourly.Variables(1).ValuesAsNumpy(),
        "pressure": hourly.Variables(2).ValuesAsNumpy(),
        # "surface_pressure": hourly.Variables(3).ValuesAsNumpy(), # Redundant
        "wind_speed": hourly.Variables(4).ValuesAsNumpy(),
        "wind_deg": hourly.Variables(5).ValuesAsNumpy(),
        "clouds": hourly.Variables(6).ValuesAsNumpy()
    }
    weather_df = pd.DataFrame(data = weather_data)

    # Process AQI Data
    response = aqi_responses[0]
    hourly = response.Hourly()
    
    hourly_data = {"datetime": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}
    
    aqi_data = {
        "datetime": hourly_data["datetime"],
        "pm10": hourly.Variables(0).ValuesAsNumpy(),
        "pm2_5": hourly.Variables(1).ValuesAsNumpy(),
        "co": hourly.Variables(2).ValuesAsNumpy(),
        "no2": hourly.Variables(3).ValuesAsNumpy(),
        "so2": hourly.Variables(4).ValuesAsNumpy(),
        "o3": hourly.Variables(5).ValuesAsNumpy(),
        "nh3": hourly.Variables(6).ValuesAsNumpy(),
        "aqi": hourly.Variables(7).ValuesAsNumpy(), # US AQI
    }
    aqi_df = pd.DataFrame(data = aqi_data)

    # Merge DataFrames
    print("Weather DF Head:")
    print(weather_df.head())
    print("AQI DF Head:")
    print(aqi_df.head())
    
    merged_df = pd.merge(weather_df, aqi_df, on="datetime")
    
    # Handle missing values
    print("Missing values before filling:")
    print(merged_df.isna().sum())
    
    merged_df = merged_df.ffill().bfill().fillna(0)
    
    # Feature Engineering
    merged_df = engineer_features(merged_df)
    
    return merged_df

def backfill_feature_store():
    if not HOPSWORKS_API_KEY:
        print("API keys missing.")
        return

    # Backfill from 2023 to 2026 (3 Years)
    start_date = "2023-01-01"
    end_date = "2026-01-31"

    df = fetch_historical_data(start_date, end_date)
    print("Data fetched successfully:")
    print(df.head())
    print(f"Total rows: {len(df)}")

    print("Connecting to Hopsworks...")
    project = hopsworks.login(project=HOPSWORKS_PROJECT_NAME, api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()

    # Define Feature Group
    try:
        # Delete if exists to ensure schema consistency
        try:
            fg = fs.get_feature_group(name="aqi_features", version=1)
            fg.delete()
            print("Deleted existing feature group.")
        except:
            pass
            
        aqi_fg = fs.get_or_create_feature_group(
            name="aqi_features",
            version=1,
            primary_key=["unix_time"],
            description="AQI and Weather features",
            online_enabled=False,
            event_time="datetime"
        )
        
        # Insert data
        print("Inserting data into Feature Store (this might take a minute)...")
        aqi_fg.insert(df)
        print("Data insertion complete.")
        
    except Exception as e:
        print(f"Error interacting with Hopsworks: {e}")

if __name__ == "__main__":
    backfill_feature_store()
