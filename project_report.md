# Project Report: Serverless Lahore AQI Prediction System

## 1. Project Overview

This project presents a **robust, serverless, end-to-end Machine Learning system** for predicting the **Air Quality Index (AQI)** of **Lahore, Pakistan** for the next **72 hours (3 days)**. The system integrates **real-time environmental data ingestion**, **automated feature engineering**, **continuous model training**, and **interactive visualization** to provide accurate and actionable air quality forecasts.

The system leverages modern **MLOps practices**, including:

-   **Feature Stores and Model Registries** (Hopsworks)
-   **Serverless automation** via GitHub Actions
-   **Real-time dashboards** using Streamlit
-   **Multi-model training and evaluation pipelines**

The final solution is **fully automated, scalable, fault-tolerant, and production-grade**, demonstrating a complete **ML lifecycle implementation**.

---

## 2. Project Evolution & Motivation (From PM2.5 to AQI – A Complete Reset)

### 2.1 Original Project Scope

The initial version of this project was designed to **predict PM2.5 concentration levels** using historical weather and pollutant data. The core components included:

-   Feature ingestion pipelines
-   LSTM-based deep learning models
-   Real-time data retrieval using OpenWeather APIs
-   Dashboard-based visualization

### 2.2 Transition from PM2.5 to AQI

During development, the scope was strategically **expanded from PM2.5 prediction to full AQI prediction**, since AQI:

-   Is a **composite metric**, integrating multiple pollutants
-   Provides **better real-world interpretability**
-   Aligns with **public health standards** (US EPA AQI scale)

This transition required:

-   Reengineering feature pipelines
-   Implementing AQI standardization logic
-   Modifying model targets and training workflows

### 2.3 Critical System Failure & Decision to Restart

During the migration phase, **accidental deletion of core project files** resulted in:

-   Complete pipeline breakdown
-   Inconsistent feature definitions
-   Corrupted training scripts
-   Incompatible model artifacts

Given the tight coupling between **Feature engineering**, **Model training**, and **Inference pipelines**, continuing development on the damaged codebase became **unreliable and inefficient**.

#### Strategic Decision: Full System Reset

Rather than patching unstable legacy code, a **new repository was initialized**, marking the **start of a clean, redesigned architecture**, with:

-   Proper modular separation
-   Robust automation pipelines
-   Production-grade MLOps practices
-   Full AQI standardization

This restart resulted in the **current system**, which is significantly **more stable, scalable, and maintainable** than the original implementation.

---

## 3. Key Objectives

-   Predict AQI for the **next 72 hours**
-   Build **serverless ML pipelines**
-   Implement **automated feature engineering & training**
-   Integrate **feature store and model registry**
-   Provide **real-time, explainable predictions**
-   Ensure **production-grade fault tolerance**

---

## 4. Technical Architecture & Tech Stack

### 4.1 Tech Stack

| Component | Technology |
| :--- | :--- |
| **Language** | Python 3.9+ |
| **Data Processing** | Pandas, NumPy |
| **ML Frameworks** | Scikit-learn, XGBoost, TensorFlow (Keras) |
| **Feature Store & Model Registry** | Hopsworks |
| **Orchestration** | GitHub Actions |
| **APIs** | OpenWeather, Open-Meteo |
| **Frontend** | Streamlit |
| **Explainability** | SHAP |

### 4.2 High-Level System Architecture

The system follows a three-layer serverless ML pipeline architecture:

```mermaid
graph LR
    A[APIs] --> B[Feature Pipeline]
    B --> C[Hopsworks Feature Store]
    C --> D[Training Pipeline]
    D --> E[Model Registry]
    E --> F[Streamlit App]
```

---

## 5. Pipeline Architecture & Workflow

### 5.1 Feature Pipeline (Hourly)

-   **Fetches** real-time weather & pollutant data
-   **Computes:**
    -   Time features (`hour`, `day`, `month`, `weekday`)
    -   Lag features (`aqi_prev_1h`, `aqi_change_1h`, `aqi_lag_24h`)
    -   AQI standardization (US EPA 0–500 scale)
-   **Stores** processed features into **Hopsworks Feature Store**
-   **Automation:** GitHub Actions – `feature_pipeline.yml` (hourly execution)

### 5.2 Training Pipeline (Daily)

-   **Retrieves** training data via Feature Views
-   **Trains** multiple candidate models:
    -   Ridge Regression
    -   Random Forest
    -   XGBoost
    -   LSTM (Deep Learning)
-   **Evaluates** models using:
    -   RMSE
    -   MAE
    -   R²
-   **Registers** best-performing model to **Hopsworks Model Registry**
-   **Automation:** GitHub Actions – `training_pipeline.yml` (daily execution)

### 5.3 Inference & Dashboard Pipeline (On-Demand)

-   **Loads** latest model
-   **Fetches** 72-hour weather forecast
-   **Generates** AQI predictions
-   **Displays:**
    -   Real-time AQI
    -   72-hour forecast
    -   Model comparison
    -   SHAP-based explainability
-   **Interface:** Streamlit Web Application

---

## 6. Machine Learning Model Design

### 6.1 Problem Formulation

-   **Type:** Multistep Time Series Regression
-   **Inputs:** Weather + pollutant + engineered features
-   **Outputs:** AQI values for next 72 hours

### 6.2 Model Architectures

| Model | Purpose |
| :--- | :--- |
| **Ridge Regression** | Linear baseline |
| **Random Forest** | Non-linear tree ensemble |
| **XGBoost** | Gradient boosting |
| **LSTM** | Deep time-series forecasting |

### 6.3 Why LSTM?

LSTM effectively:
-   Captures **long-term temporal dependencies**
-   Learns **pollution accumulation patterns**
-   Models **seasonal and cyclic behavior**

---

## 7. Implementation Journey

### Phase 1: Data & Feature Engineering
-   Integrated **OpenWeather** (live) and **Open-Meteo** (historical)
-   Designed:
    -   Temporal features
    -   Lagged AQI features
    -   Composite primary keys in Feature Store
-   Ensured **training–inference feature consistency**

### Phase 2: Model Training & Registry
-   Implemented **multi-model training pipeline**
-   Automated **model evaluation**
-   Implemented **best-model selection logic**
-   Stored:
    -   Model artifacts
    -   Scalers
    -   Metrics JSON (for full lineage tracking)

### Phase 3: Dashboard & Deployment
-   Designed professional **Streamlit UI**
-   Added:
    -   Forecast charts
    -   Real-time metrics
    -   SHAP explainability
-   Deployed using **Streamlit Cloud** with secure secrets handling

---

## 8. Challenges Encountered & Solutions

### 8.1 Full Project Reset
-   **Problem:** Accidental deletion of core PM2.5 pipeline caused system-wide failure.
-   **Solution:** Restarted entire architecture from scratch using:
    -   Modular pipelines
    -   Clean architecture
    -   Strong separation of concerns

### 8.2 Hopsworks Connectivity Issues (Windows)
-   **Problem:** Arrow Flight protocol instability, SSL failures, socket crashes.
-   **Solution:** Implemented:
    -   SQL fallback access
    -   Dataset materialization strategy
    -   Runtime monkeypatching for SSL compatibility

### 8.3 Feature Ordering Crash
-   **Problem:** `ValueError: Feature names must be in the same order`
-   **Solution:** Dynamic feature reordering using:
    -   `model.feature_names_in_`
    -   `model.get_booster().feature_names`

### 8.4 Streamlit Cloud Secrets Handling
-   **Problem:** API keys not detected in cloud.
-   **Solution:** Dual environment logic using:
    -   `os.getenv()`
    -   `st.secrets`

### 8.5 Missing Dashboard Metrics
-   **Problem:** Empty sidebar when registry metadata unavailable.
-   **Solution:** JSON fallback mechanism for metrics loading.

---

## 9. Key Learnings

1.  **MLOps > ML:** Only 20% effort was model building; 80% was pipelines, orchestration, reliability.
2.  **Feature Stores:** Decoupled training & inference pipelines.
3.  **Serverless:** GitHub Actions eliminated infrastructure costs.
4.  **Defensive Engineering:** Production systems require fallback strategies.
5.  **Environment Management:** Multi-platform dependency handling is critical.

---

## 10. Current System Status

The current implementation is:
-   **Fully automated**
-   **Serverless**
-   **Multi-model**
-   **Feature-store integrated**
-   **Cloud deployable**
-   **Production-ready**

The architecture is **stable, scalable, and maintainable**, providing accurate **real-time AQI forecasting**.

---

## 11. Future Improvements

-   **Drift detection & monitoring**
-   **Transformer-based forecasting models**
-   **Multi-city AQI prediction**
-   **Health advisory alerts**
-   **Satellite + traffic data integration**

---

## 12. Conclusion

This project successfully demonstrates the design and implementation of a complete **end-to-end AQI prediction system** using modern **MLOps** and **serverless cloud architecture**.

By restarting from a failed PM2.5 system and rebuilding the pipeline with a clean, modular design, the final solution achieves **robust automation, high reliability, scalability, and real-world usability**.

The project stands as a **production-grade ML system**, not just a research prototype — making it suitable for academic evaluation, industrial deployment, and professional portfolios.
