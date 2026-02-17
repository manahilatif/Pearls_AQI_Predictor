# 🌫️ Lahore AQI Predictor

A robust, end-to-end Machine Learning pipeline and interactive dashboard to forecast the Air Quality Index (AQI) for Lahore, Pakistan.

## 🚀 Live Demo
*[Insert your Streamlit App Link Here]*

## 📖 Overview
Air quality in Lahore is a critical concern. This project automates the collection of weather and pollution data, trains multiple ML models (including Deep Learning), and serves predictions via a user-friendly web interface.

**Key Features:**
*   **Automated Data Pipeline:** Fetches hourly data from **Open-Meteo** and **OpenWeatherMap**.
*   **Feature Store:** Utilizes **Hopsworks** to manage historical features (Backfilled from 2023-2026) and ensure data consistency.
*   **Multi-Model Training:** Trains and evaluates **Random Forest**, **Ridge Regression**, **XGBoost**, and **LSTM** (Deep Learning) models daily.
*   **Interactive Dashboard:** Built with **Streamlit**, featuring:
    *   Real-time 3-day AQI Forecasts.
    *   Historical Trends Visualization.
    *   **EDA:** Correlation Heatmaps.
    *   **Explainable AI:** SHAP plots to understand feature impact.
*   **CI/CD:** Fully automated via **GitHub Actions** (Hourly Feature Pipeline & Daily Training Pipeline).
*   **Robustness:** Includes auto-retry logic and mock data failover for reliable local testing.

## 🛠️ Tech Stack
*   **Language:** Python 3.9
*   **Data Ops:** Hopsworks (Feature Store & Model Registry)
*   **ML Libraries:** Scikit-learn, XGBoost, TensorFlow/Keras (LSTM), SHAP
*   **Web App:** Streamlit, Seaborn, Matplotlib
*   **Automation:** GitHub Actions
*   **APIs:** Open-Meteo

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/manahilatif/pearls_aqi_predictor.git
cd pearls_aqi_predictor
```

### 2. Set up Virtual Environment
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configuration (.env)
Create a `.env` file in the root directory and add your Hopsworks credentials:
```ini
HOPSWORKS_API_KEY=your_hopsworks_api_key
HOPSWORKS_PROJECT_NAME=your_project_name
```

## 🏃‍♂️ Usage

### Running the Dashboard Locally
```bash
streamlit run app.py
```
*Note: If the Hopsworks connection fails locally (e.g., due to firewalls), the app will automatically fallback to Mock Data for demonstration purposes.*

### Running Pipelines Manually
You can trigger the pipelines locally to fetch data or retrain models:
```bash
# Run Feature Pipeline (Fetch new data)
python src/features/feature_pipeline.py

# Run Training Pipeline (Train & Register models)
python src/models/train_model.py
```

## 🤖 Automated Pipelines (GitHub Actions)
This repository includes configured workflows in `.github/workflows/`:
1.  **Feature Pipeline:** Runs **hourly** to fetch the latest weather/AQI data and update the Feature Store.
2.  **Training Pipeline:** Runs **daily** at midnight to retrain all models on the latest dataset and update the Model Registry.

## 📂 Project Structure
```
├── .github/workflows/   # CI/CD definitions
├── src/
│   ├── features/        # Data fetching, backfilling & engineering scripts
│   ├── models/          # Model training, evaluation & registration logic
├── app.py               # Streamlit Dashboard entry point
├── requirements.txt     # Python dependencies
├── runtime.txt          # Python version pinning (3.9) for Streamlit Cloud
└── README.md            # Project documentation
```

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request.

---
*Built with ❤️ for 10Pearls Shine Internship (Dec 2025 - Feb 2026)*
