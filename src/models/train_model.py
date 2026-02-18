"""
train_model.py
--------------
Fetches features from Hopsworks Feature Store, trains 4 models
(RandomForest, Ridge, XGBoost, LSTM), evaluates each on multiple
regression metrics, selects the best model, and registers ALL models
plus the best-model tag in the Hopsworks Model Registry.

Runs daily via GitHub Actions.

Usage:
    python src/models/train_model.py
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# ── TensorFlow first to avoid DLL conflicts ────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import hopsworks
import pandas as pd
import numpy as np
import joblib
from dotenv import load_dotenv

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

from hsml.schema import Schema
from hsml.model_schema import ModelSchema

load_dotenv()

HOPSWORKS_API_KEY      = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")


# ══════════════════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(y_true, y_pred, model_name):
    """
    Pure regression metrics only.
    - RMSE  : penalises large errors heavily (primary comparison metric)
    - MAE   : average absolute error in AQI units (interpretable)
    - R²    : proportion of variance explained (higher = better)
    - MAPE  : percentage error (scale-independent)
    - SMAPE : symmetric MAPE (handles near-zero values better)

    NO classification metrics — binning continuous predictions into
    categories introduces rounding artefacts and is misleading for a
    regression task.
    """
    y_true_np = np.array(y_true).flatten().astype(float)
    y_pred_np = np.array(y_pred).flatten().astype(float)

    rmse  = float(np.sqrt(mean_squared_error(y_true_np, y_pred_np)))
    mae   = float(mean_absolute_error(y_true_np, y_pred_np))
    r2    = float(r2_score(y_true_np, y_pred_np))
    mape  = float(np.mean(np.abs((y_true_np - y_pred_np) / (y_true_np + 1e-8))) * 100)
    smape = float(np.mean(
        2 * np.abs(y_true_np - y_pred_np) /
        (np.abs(y_true_np) + np.abs(y_pred_np) + 1e-8)
    ) * 100)

    # Per-horizon breakdown (Day 1 / Day 2 / Day 3)
    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    horizon_rmse = {}
    if y_t.ndim == 2 and y_t.shape[1] == 3:
        labels = ["Day1", "Day2", "Day3"]
        for i, label in enumerate(labels):
            h_rmse = float(np.sqrt(mean_squared_error(y_t[:, i], y_p[:, i])))
            horizon_rmse[f"RMSE_{label}"] = round(h_rmse, 4)

    print(f"\n{'='*50}")
    print(f"  {model_name} Evaluation")
    print(f"{'='*50}")
    print(f"  RMSE  : {rmse:.4f}  ← primary ranking metric")
    print(f"  MAE   : {mae:.4f}")
    print(f"  R²    : {r2:.4f}")
    print(f"  MAPE  : {mape:.2f}%")
    print(f"  SMAPE : {smape:.2f}%")
    if horizon_rmse:
        for k, v in horizon_rmse.items():
            print(f"  {k}  : {v:.4f}")
    print(f"{'='*50}")

    metrics = {
        "RMSE":  round(rmse,  4),
        "MAE":   round(mae,   4),
        "R2":    round(r2,    4),
        "MAPE":  round(mape,  2),
        "SMAPE": round(smape, 2),
        **horizon_rmse
    }
    return metrics


def select_best_model(results: dict) -> str:
    """
    Ranks all trained models and returns the name of the best one.

    Composite score (lower = better):
        score = 0.50 * norm_RMSE
              + 0.25 * norm_MAE
              + 0.15 * norm_MAPE
              + 0.10 * (1 - norm_R2)   ← penalise low R²

    Normalises each metric to [0, 1] across models so no single metric
    dominates due to scale differences.
    """
    names  = list(results.keys())
    rmses  = np.array([results[n]["RMSE"]  for n in names], dtype=float)
    maes   = np.array([results[n]["MAE"]   for n in names], dtype=float)
    mapes  = np.array([results[n]["MAPE"]  for n in names], dtype=float)
    r2s    = np.array([results[n]["R2"]    for n in names], dtype=float)

    def norm(arr):
        rng = arr.max() - arr.min()
        return (arr - arr.min()) / (rng + 1e-8)

    scores = (
        0.50 * norm(rmses) +
        0.25 * norm(maes)  +
        0.15 * norm(mapes) +
        0.10 * (1 - norm(r2s))
    )

    print("\n📊 Model Comparison (lower composite score = better):")
    print(f"  {'Model':<15} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'MAPE':>8} {'Score':>8}")
    print(f"  {'-'*55}")
    for i, name in enumerate(names):
        m = results[name]
        print(f"  {name:<15} {m['RMSE']:>8.4f} {m['MAE']:>8.4f} "
              f"{m['R2']:>8.4f} {m['MAPE']:>7.2f}% {scores[i]:>8.4f}")

    best_idx  = int(np.argmin(scores))
    best_name = names[best_idx]
    print(f"\n🏆 Best model: {best_name} (composite score: {scores[best_idx]:.4f})")
    return best_name


# ══════════════════════════════════════════════════════════════════════════════
#  LSTM HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def create_sequences(X: np.ndarray, y: np.ndarray, window: int = 7):
    """
    Builds (samples, window, features) sequences for LSTM.
    Uses a 7-day rolling window so the model learns temporal patterns
    rather than treating each day independently.
    """
    Xs, ys = [], []
    for i in range(window, len(X)):
        Xs.append(X[i - window:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def build_lstm_model(window: int, n_features: int, n_outputs: int):
    model = Sequential([
        LSTM(64, activation='relu',
             input_shape=(window, n_features),
             return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation='relu', return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(n_outputs)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.summary()
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def train_and_evaluate():
    if not HOPSWORKS_API_KEY:
        print("ERROR: HOPSWORKS_API_KEY not set.")
        sys.exit(1)

    # ── Connect ────────────────────────────────────────────────────────────
    print("Connecting to Hopsworks...")
    project = hopsworks.login(
        project=HOPSWORKS_PROJECT_NAME,
        api_key_value=HOPSWORKS_API_KEY
    )
    fs = project.get_feature_store()
    mr = project.get_model_registry()
    print("Connected.")

    # ── Fetch features ─────────────────────────────────────────────────────
    print("\nFetching features from Feature Store...")
    aqi_fg = fs.get_feature_group(name="aqi_features", version=1)
    try:
        df = aqi_fg.select_all().read()
    except Exception:
        print("Arrow Flight failed, falling back to Hive...")
        df = aqi_fg.select_all().read(read_options={"use_hive": True})

    print(f"Raw data shape: {df.shape}")

    # ── Prepare daily data ─────────────────────────────────────────────────
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").set_index("datetime")

    print("Resampling hourly → daily averages...")
    df_daily = df.resample("D").mean().ffill().fillna(0)

    # Targets: next 1 / 2 / 3 days AQI
    df_daily["aqi_next_day"]    = df_daily["aqi"].shift(-1)
    df_daily["aqi_next_2_days"] = df_daily["aqi"].shift(-2)
    df_daily["aqi_next_3_days"] = df_daily["aqi"].shift(-3)
    df_daily.dropna(inplace=True)

    TARGET_COLS  = ["aqi_next_day", "aqi_next_2_days", "aqi_next_3_days"]
    EXCLUDE_COLS = TARGET_COLS + ["unix_time"]
    feature_cols = [c for c in df_daily.columns if c not in EXCLUDE_COLS]

    X = df_daily[feature_cols]
    y = df_daily[TARGET_COLS]

    print(f"Training samples: {len(X)}  |  Features: {len(feature_cols)}")

    if len(X) < 30:
        print("ERROR: Need at least 30 days of data. Run backfill first.")
        sys.exit(1)

    # Time-based split (no shuffle — preserves temporal order)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    print(f"Train: {len(X_train)} days  |  Test: {len(X_test)} days")

    # ── Scaling (needed for Ridge + LSTM) ──────────────────────────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Save scaler — app.py needs it at inference time
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved → {scaler_path}")

    # Shared Hopsworks schema (same input/output for all models)
    input_schema  = Schema(X_train)
    output_schema = Schema(y_train)
    model_schema  = ModelSchema(input_schema=input_schema, output_schema=output_schema)

    all_metrics = {}   # Collects results for comparison

    # ══════════════════════════════════════════════════════════════════════
    #  SKLEARN MODELS
    # ══════════════════════════════════════════════════════════════════════
    sklearn_configs = {
        "RandomForest": {
            "estimator": RandomForestRegressor(
                n_estimators=200, random_state=42, n_jobs=-1
            ),
            "use_scaled": False,   # Tree-based models don't need scaling
        },
        "Ridge": {
            "estimator": Ridge(alpha=1.0),
            "use_scaled": True,    # Linear model needs scaling
        },
        "XGBoost": {
            "estimator": MultiOutputRegressor(
                xgb.XGBRegressor(
                    objective="reg:squarederror",
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=6,
                    random_state=42,
                    n_jobs=-1
                )
            ),
            "use_scaled": False,
        },
    }

    for name, cfg in sklearn_configs.items():
        print(f"\n{'─'*50}")
        print(f"Training {name}...")

        estimator  = cfg["estimator"]
        Xtr = X_train_sc if cfg["use_scaled"] else X_train.values
        Xte = X_test_sc  if cfg["use_scaled"] else X_test.values

        estimator.fit(Xtr, y_train)
        y_pred = estimator.predict(Xte)

        metrics = evaluate_model(y_test, y_pred, name)
        all_metrics[name] = metrics

        # Save to its own subdirectory (Hopsworks needs a directory, not a file)
        m_dir = os.path.join(model_dir, name)
        os.makedirs(m_dir, exist_ok=True)
        pkl_path = os.path.join(m_dir, f"{name}_model.pkl")
        joblib.dump(estimator, pkl_path)

        # Also save the scaler inside each model dir so app.py always finds it
        joblib.dump(scaler, os.path.join(m_dir, "scaler.pkl"))

        print(f"Registering {name} in Model Registry...")
        hs_model = mr.python.create_model(
            name=f"aqi_{name.lower()}_model",
            description=f"AQI 3-day predictor — {name} (Lahore)",
            input_example=X_train.sample(1),
            model_schema=model_schema,
            metrics=metrics
        )
        hs_model.save(m_dir)   # ✅ directory path
        print(f"{name} registered ✓")

    # ══════════════════════════════════════════════════════════════════════
    #  LSTM
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*50}")
    print("Training LSTM (7-day sequence window)...")

    WINDOW = 7
    X_seq_train, y_seq_train = create_sequences(X_train_sc, y_train.values, WINDOW)
    X_seq_test,  y_seq_test  = create_sequences(X_test_sc,  y_test.values,  WINDOW)

    if len(X_seq_train) < 10:
        print("WARNING: Not enough data for LSTM sequences. Skipping.")
    else:
        lstm = build_lstm_model(
            window=WINDOW,
            n_features=X_seq_train.shape[2],
            n_outputs=3
        )

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1
        )

        lstm.fit(
            X_seq_train, y_seq_train,
            epochs=150,
            batch_size=16,
            validation_split=0.15,
            callbacks=[early_stop],
            verbose=1
        )

        y_pred_lstm = lstm.predict(X_seq_test)
        metrics_lstm = evaluate_model(y_seq_test, y_pred_lstm, "LSTM")
        all_metrics["LSTM"] = metrics_lstm

        # Save LSTM to its own directory
        lstm_dir = os.path.join(model_dir, "LSTM")
        os.makedirs(lstm_dir, exist_ok=True)
        lstm.save(os.path.join(lstm_dir, "lstm_model.keras"))
        joblib.dump(scaler, os.path.join(lstm_dir, "scaler.pkl"))

        print("Registering LSTM in Model Registry...")
        hs_lstm = mr.python.create_model(
            name="aqi_lstm_model",
            description="AQI 3-day predictor — LSTM 7-day window (Lahore)",
            input_example=X_train.sample(1),
            model_schema=model_schema,
            metrics=metrics_lstm
        )
        hs_lstm.save(lstm_dir)   # ✅ directory path
        print("LSTM registered ✓")

    # ══════════════════════════════════════════════════════════════════════
    #  MODEL COMPARISON & BEST MODEL SELECTION
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*50}")
    print("  MODEL COMPARISON")
    print(f"{'═'*50}")

    best_model_name = select_best_model(all_metrics)

    # Store best model name + full comparison table in Hopsworks
    # as a lightweight "metadata model" so app.py can read it
    comparison_df = pd.DataFrame(all_metrics).T.reset_index()
    comparison_df.columns = ["model"] + list(comparison_df.columns[1:])
    comparison_df["is_best"] = comparison_df["model"] == best_model_name

    print("\nSaving comparison table to Feature Store...")
    try:
        comp_fg = fs.get_or_create_feature_group(
            name="model_comparison",
            version=1,
            primary_key=["model"],
            description="Model evaluation metrics and best model flag",
            online_enabled=True
        )
        comp_fg.insert(comparison_df)
        print("Comparison table saved ✓")
    except Exception as e:
        print(f"Could not save comparison table: {e}")

    print(f"\n✅ All models trained and registered.")
    print(f"🏆 Best model: {best_model_name}")
    print(f"   → Hopsworks name: aqi_{best_model_name.lower()}_model")


if __name__ == "__main__":
    train_and_evaluate()