# AIRSIGHT AI 🌍

### AI-Powered Multi-City Air Quality Monitoring, Prediction & Early Warning System

AIRSIGHT AI is a data science and machine learning project designed to collect, analyze, and predict air quality across major Indian cities. The goal is to transform environmental data into meaningful insights and early warnings that can help people understand changing air pollution conditions.

## 🎯 Project Objectives

- Collect air quality data from multiple Indian cities.
- Monitor pollutants such as PM2.5, PM10, NO₂ and other available parameters.
- Perform exploratory data analysis and identify pollution trends.
- Develop machine learning models for air quality forecasting.
- Build an interactive dashboard for visualizing air quality.
- Explore an early-warning system for potentially poor air quality.

## 📍 Target Cities

1. Delhi
2. Mumbai
3. Ahmedabad
4. Bengaluru
5. Chennai
6. Kolkata
7. Hyderabad
8. Pune

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data processing | Pandas, NumPy |
| Data visualization | Matplotlib, Seaborn, Plotly |
| Machine learning | Scikit-learn, XGBoost (planned) |
| Deep learning | LSTM (planned) |
| Dashboard | Streamlit |
| Data source | OpenAQ API |
| Version control | Git and GitHub |

## 📊 Current Development Progress

- [x] Initial project structure
- [x] Python environment setup
- [x] OpenAQ API authentication
- [x] API connectivity test
- [x] Initial multi-city location collection
- [ ] Sensor discovery and metadata collection
- [ ] Pollutant measurement collection
- [ ] Data cleaning and validation
- [ ] Exploratory data analysis
- [ ] Machine learning model development
- [ ] Air quality forecasting
- [ ] Interactive dashboard
- [ ] Early-warning system
- [ ] Deployment

### Initial Data Collection Test

The first collection test retrieved 244 monitoring-location records across eight target cities.

| City | Records retrieved |
|---|---:|
| Delhi | 97 |
| Mumbai | 44 |
| Bengaluru | 26 |
| Pune | 19 |
| Kolkata | 19 |
| Ahmedabad | 14 |
| Hyderabad | 14 |
| Chennai | 11 |
| **Total** | **244** |

These are initial API search results, not a verified count of all available stations. Coverage, pagination, sensors and measurement availability still need to be validated.

## 📁 Project Structure

```text
AIRSIGHT-AI/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── notebooks/
├── src/
│   ├── data_collection/
│   ├── config.py
│   └── test_api.py
│
├── dashboard/
├── models/
├── reports/
├── logs/
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/AIRSIGHT-AI.git
cd AIRSIGHT-AI
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the API key

Create a `.env` file in the project root:

```env
OPENAQ_API_KEY=your_openaq_api_key
```

Never commit your actual `.env` file or API key.

### 5. Run the API connection test

```bash
python -m src.test_api
```

## 🌐 Data Source

Air-quality monitoring data is collected through the [OpenAQ API](https://openaq.org/).

API access requires an OpenAQ API key. Sensor and measurement availability depends on the monitoring locations and data providers.

## 🚀 Future Scope

- Multi-city air quality forecasting.
- Time-series prediction using machine learning and deep learning.
- Interactive geospatial pollution visualization.
- Pollution trend analysis and city comparisons.
- Automated early warnings based on validated thresholds.
- Cloud deployment and API integration.

## 👨‍💻 Author

**Yash Darji**

Computer Science Engineering — Artificial Intelligence & Data Science

## 📌 Project Status

Currently under active development.

*This project is being developed as a practical exploration of data engineering, environmental analytics, and machine learning.*