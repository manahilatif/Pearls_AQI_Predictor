import os
import sys
# Add 'src' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import hopsworks
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime

# Load environment variables
load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")

def train_and_evaluate():
    if not HOPSWORKS_API_KEY:
        print("API keys missing.")
        return

    print("Connecting to Hopsworks...")
    project = hopsworks.login(project=HOPSWORKS_PROJECT_NAME, api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()

    try:
        # Fetch data
        print("Fetching data from Feature Store...")
        aqi_fg = fs.get_feature_group(name="aqi_features", version=1)
        query = aqi_fg.select_all()
        # Use Hive as fallback if Query Service acts up
        try:
             df = query.read()
        except:
             print("Arrow Flight failed, trying Hive...")
             try:
                 df = query.read(read_options={"use_hive": True})
             except:
                 print("Hive failed, trying local cache...")
                 if os.path.exists("data/processed/aqi_history.csv"):
                     df = pd.read_csv("data/processed/aqi_history.csv")
                     df['datetime'] = pd.to_datetime(df['datetime'])
                 else:
                     raise Exception("Could not read from Feature Store or local cache.")
        
        # Sort by time
        df = df.sort_values(by="datetime")
        df.set_index("datetime", inplace=True)
        
        # Resample to Daily average for 3-day prediction
        print("Resampling to daily averages...")
        df_daily = df.resample('D').mean()
        df_daily.dropna(inplace=True)
        
        # Create Targets (Next 3 days)
        df_daily['aqi_next_day'] = df_daily['aqi'].shift(-1)
        df_daily['aqi_next_2_days'] = df_daily['aqi'].shift(-2)
        df_daily['aqi_next_3_days'] = df_daily['aqi'].shift(-3)
        
        # Drop last 3 rows (NaN targets)
        df_daily.dropna(inplace=True)
        
        print(f"Training data shape: {df_daily.shape}")
        
        # Features and Targets
        # Use simple features for now (current day values)
        X = df_daily.drop(columns=['aqi_next_day', 'aqi_next_2_days', 'aqi_next_3_days', 'unix_time'])
        y = df_daily[['aqi_next_day', 'aqi_next_2_days', 'aqi_next_3_days']]
        
        # Split (Time-based, no shuffle)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        print("Training models...")
        models = {
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Ridge": Ridge(alpha=1.0),
            "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
        }
        
        results = {}
        best_model_name = None
        best_rmse = float('inf')
        best_model_obj = None

        # Train and Evaluate for each target horizon? 
        # For simplicity, default sklearn regressors support multi-output.
        # XGBoost supports multi-output via wrapper or natively? 
        # XGBRegressor usually supports single output, but recent versions support MultiOutputRegressor automatically or natively.
        # Let's use MultiOutputRegressor wrapper if needed, but RF and Ridge handle it.
        # Check XGBoost multi-output support.
        
        from sklearn.multioutput import MultiOutputRegressor
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            if name == "XGBoost":
                 # XGBoost Regressor handles multi-output if configured? Or wrap it.
                 # Using wrapper is safer.
                 estimator = MultiOutputRegressor(model)
            else:
                 estimator = model
            
            estimator.fit(X_train, y_train)
            y_pred = estimator.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2}
            print(f"{name} Results: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_model_name = name
                best_model_obj = estimator
        
        print(f"\nBest Model: {best_model_name} with RMSE: {best_rmse:.4f}")
        
        # Save Best Model to Registry
        mr = project.get_model_registry()
        
        # Create a local dir for artifacts
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, "best_aqi_model.pkl")
        joblib.dump(best_model_obj, model_path)
        
        
        # Input/Output schema
        from hsml.schema import Schema
        from hsml.model_schema import ModelSchema
        
        input_schema = Schema(X_train)
        output_schema = Schema(y_train)
        model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)
        
        aqi_model = mr.python.create_model(
            name="aqi_predictor_model",
            description=f"Best AQI Predictor ({best_model_name})",
            input_example=X_train.sample(1),
            model_schema=model_schema,
            metrics={"RMSE": best_rmse, "R2": results[best_model_name]["R2"]}
        )
        
        aqi_model.save(model_path)
        print("Model registered in Hopsworks Model Registry.")
        
    except Exception as e:
        print(f"Error in training pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_and_evaluate()
