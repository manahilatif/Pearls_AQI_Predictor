import sys
import os
# Add 'src' to sys.path to allow imports from features module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import hopsworks
import pandas as pd
from dotenv import load_dotenv
from src.features.utils import fetch_weather_data, process_data

# Load environment variables
load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Default location (can be made configurable)
LAT = 51.5074 # London
LON = -0.1278

def run_feature_pipeline():
    if not HOPSWORKS_API_KEY or not OPENWEATHER_API_KEY:
        print("Error: API keys not found. Please set HOPSWORKS_API_KEY and OPENWEATHER_API_KEY in .env file.")
        return

    print("Fetching data...")
    try:
        weather_data, aqi_data = fetch_weather_data(LAT, LON, OPENWEATHER_API_KEY)
        df = process_data(weather_data, aqi_data)
        print("Data fetched successfully:")
        print(df.head())
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    print("Connecting to Hopsworks...")
    project = hopsworks.login(project=HOPSWORKS_PROJECT_NAME, api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()

    # Define Feature Group
    try:
        aqi_fg = fs.get_or_create_feature_group(
            name="aqi_features",
            version=1,
            primary_key=["datetime"],
            description="AQI and Weather features"
        )
        print("Feature Group retrieved/created.")
        
        # Insert data
        aqi_fg.insert(df)
        print("Data inserted into Feature Group.")
        
    except Exception as e:
        print(f"Error interacting with Hopsworks: {e}")

if __name__ == "__main__":
    run_feature_pipeline()
