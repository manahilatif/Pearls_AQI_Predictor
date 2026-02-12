import os
import sys
# Add 'src' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import TensorFlow first to avoid DLL conflicts
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

import hopsworks
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score, accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
import joblib
import json

from hsml.schema import Schema
from hsml.model_schema import ModelSchema

# Load environment variables
load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")

def get_aqi_category(aqi_value):
    if aqi_value <= 50: return 0 # Good
    elif aqi_value <= 100: return 1 # Moderate
    elif aqi_value <= 150: return 2 # Unhealthy for Sensitive Groups
    elif aqi_value <= 200: return 3 # Unhealthy
    elif aqi_value <= 300: return 4 # Very Unhealthy
    else: return 5 # Hazardous

def evaluate_models(y_true, y_pred, model_name):
    # Regression Metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Classification Metrics (Binning)
    y_true_bins = np.array([[get_aqi_category(val) for val in row] for row in y_true.to_numpy()])
    y_pred_bins = np.array([[get_aqi_category(val) for val in row] for row in y_pred])
    
    # Flatten for overall metrics across all horizons
    y_true_flat = y_true_bins.flatten()
    y_pred_flat = y_pred_bins.flatten()
    
    accuracy = accuracy_score(y_true_flat, y_pred_flat)
    precision = precision_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0)
    f1 = f1_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0)
    
    print(f"\n{model_name} Metrics:")
    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}, MSE: {mse:.4f}")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}")
    
    return {
        "RMSE": rmse, "MAE": mae, "R2": r2, "MSE": mse,
        "Accuracy": accuracy, "Precision": precision, "F1": f1
    }

def create_lstm_model(input_shape, output_shape):
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=input_shape, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(output_shape)) # Output 3 values
    model.compile(optimizer='adam', loss='mse')
    return model

def train_and_evaluate():
    if not HOPSWORKS_API_KEY:
        print("API keys missing.")
        return

    print("Connecting to Hopsworks...")
    project = hopsworks.login(project=HOPSWORKS_PROJECT_NAME, api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    try:
        # Fetch data
        print("Fetching data from Feature Store...")
        aqi_fg = fs.get_feature_group(name="aqi_features", version=1)
        query = aqi_fg.select_all()
        
        try:
             df = query.read()
        except:
             print("Arrow Flight failed, trying Hive...")
             df = query.read(read_options={"use_hive": True})
        
        # Sort by time
        df = df.sort_values(by="datetime")
        df.set_index("datetime", inplace=True)
        
        # Resample to Daily average
        print("Resampling to daily averages...")
        df_daily = df.resample('D').mean()
        # Handle missing values if any
        df_daily = df_daily.ffill().fillna(0)
        
        # Create Targets (Next 3 days)
        df_daily['aqi_next_day'] = df_daily['aqi'].shift(-1)
        df_daily['aqi_next_2_days'] = df_daily['aqi'].shift(-2)
        df_daily['aqi_next_3_days'] = df_daily['aqi'].shift(-3)
        
        # Drop last 3 rows (NaN targets)
        df_daily.dropna(inplace=True)
        
        print(f"Training data shape: {df_daily.shape}")
        
        # Features and Targets
        feature_cols = [c for c in df_daily.columns if c not in ['aqi_next_day', 'aqi_next_2_days', 'aqi_next_3_days', 'unix_time']]
        X = df_daily[feature_cols]
        y = df_daily[['aqi_next_day', 'aqi_next_2_days', 'aqi_next_3_days']]
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        
        # Models
        models = {
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Ridge": Ridge(alpha=1.0),
            "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
        }
        
        # Train and Evaluate Standard Models
        for name, model in models.items():
            print(f"\nTraining {name}...")
            if name == "XGBoost":
                 estimator = MultiOutputRegressor(model)
            else:
                 estimator = model
            
            estimator.fit(X_train, y_train)
            y_pred = estimator.predict(X_test)
            
            metrics = evaluate_models(y_test, y_pred, name)
            
            # Save and Register
            model_path = os.path.join(model_dir, f"{name}_model.pkl")
            joblib.dump(estimator, model_path)
            
            input_schema = Schema(X_train)
            output_schema = Schema(y_train)
            model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)
            
            print(f"Registering {name}...")
            hs_model = mr.python.create_model(
                name=f"aqi_{name.lower()}_model",
                description=f"AQI Predictor using {name} (Lahore)",
                input_example=X_train.sample(1),
                model_schema=model_schema,
                metrics=metrics
            )
            hs_model.save(model_path)
            
        # Train LSTM
        print("\nTraining LSTM...")
        # LSTM needs 3D input [samples, time steps, features]
        # Reshape X for LSTM (using 1 time step for simplicity, leveraging sliding window features if present)
        X_train_lstm = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test_lstm = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))
        
        lstm_model = create_lstm_model((1, X_train.shape[1]), 3)
        lstm_model.fit(X_train_lstm, y_train, epochs=50, batch_size=16, verbose=0)
        
        y_pred_lstm = lstm_model.predict(X_test_lstm)
        metrics_lstm = evaluate_models(y_test, y_pred_lstm, "LSTM")
        
        # Save and Register LSTM
        lstm_path = os.path.join(model_dir, "lstm_model")
        lstm_model.save(lstm_path)
        
        # Archive LSTM folder to zip for Hopsworks
        import shutil
        shutil.make_archive(lstm_path, 'zip', lstm_path)
        
        print("Registering LSTM...")
        # Note: Tensorflow schema support might need specific handling or just use same schema since input/output structure is same conceptually
        hs_model_lstm = mr.python.create_model(
            name="aqi_lstm_model",
            description="AQI Predictor using LSTM (Lahore)",
            input_example=X_train.sample(1), # Schema inference from dataframe
            model_schema=model_schema,
            metrics=metrics_lstm
        )
        hs_model_lstm.save(f"{lstm_path}.zip")

        print("All models trained and registered.")

    except Exception as e:
        print(f"Error in training pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_and_evaluate()
