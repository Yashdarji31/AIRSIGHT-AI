# AIRSIGHT AI — Final Hybrid Model Evaluation Report

## 1. Project Overview

AIRSIGHT AI is a multi-city air-quality monitoring and next-hour pollutant prediction project covering eight Indian cities:

- Ahmedabad
- Bengaluru
- Chennai
- Delhi
- Hyderabad
- Kolkata
- Mumbai
- Pune

The system monitors and predicts six pollutants:

- PM2.5
- PM10
- NO2
- SO2
- CO
- O3

The project implements a hybrid prediction architecture that selects between machine learning models and a persistence baseline based on validation performance.

**Prediction horizon:** One hour ahead.

**Important:** The system predicts individual pollutant concentrations. It does not currently calculate a validated Air Quality Index (AQI).

## 2. Dataset Overview

| Metric | Value |
|---|---:|
| Cities | 8 |
| Selected sensors | 48 |
| Pollutants | 6 |
| Cleaned hourly measurements | 6,996 |
| Feature dataset rows | 6,863 |
| PM2.5 test rows | 261 |
| PM10 test rows | 251 |
| NO2 test rows | 256 |
| SO2 test rows | 246 |
| CO test rows | 256 |
| O3 test rows | 256 |
| Total backtesting observations | 1,526 |
| Prediction horizon | 1 hour |

One sensor was selected per city-pollutant combination.

The dataset does not represent a complete city-wide monitoring network. Therefore, predictions should be interpreted as sensor-specific results rather than city-wide pollution estimates.

The available historical collection period is approximately seven days, which limits the ability to establish long-term forecasting performance.

## 3. Data Processing and Feature Engineering

The data processing pipeline includes:

1. OpenAQ API data collection.
2. Sensor discovery and selection.
3. Hourly measurement processing.
4. Data cleaning and quality checks.
5. Time-series feature engineering.
6. Chronological dataset preparation.

The engineered features include:

- Current observation (`lag_1h`)
- Historical observations (`lag_3h`, `lag_6h`, `lag_24h`)
- Rolling averages (`rolling_mean_3h`, `rolling_mean_6h`, `rolling_mean_24h`)
- City and sensor identifiers
- Latitude and longitude
- Hour, day of week and month

The prediction target is the next-hour pollutant concentration.

Missing historical observations are not automatically interpolated.

## 4. Model Development

The following approaches were evaluated:

1. Persistence baseline
2. Linear Regression
3. Random Forest Regressor
4. HistGradientBoosting Regressor

The ML pipelines include:

- Numerical median imputation.
- Categorical most-frequent imputation.
- One-hot encoding of categorical features.
- Feature scaling for Linear Regression.

### 4.1 Hybrid Model Selection Strategy

The hybrid architecture selects an ML model only when both validation MAE and validation RMSE improve over the persistence baseline.

Otherwise, persistence is selected.

The final model registry contains:

| Pollutant | Selected model | Training rows |
|---|---|---:|
| PM2.5 | Hist Gradient Boosting | 920 |
| PM10 | Persistence | 859 |
| NO2 | Persistence | 836 |
| SO2 | Linear Regression | 912 |
| CO | Persistence | 897 |
| O3 | Random Forest | 913 |

This architecture avoids deploying a more complicated model when it does not demonstrate sufficient validation improvement.

## 5. Hybrid Backtesting Methodology

The hybrid models were evaluated using the pollutant-specific chronological test datasets.

The evaluation compares:

- Hybrid prediction: Output from the selected model in the hybrid registry.
- Persistence prediction: The most recent observation (`lag_1h`).
- Actual value: The recorded next-hour target (`target_value`).

Both approaches are evaluated on identical observations.

The test datasets contain 1,526 usable observations across all six pollutants.

Missing model input features are handled by the trained pipelines' existing imputation steps. Rows without a valid actual target or persistence input are excluded.

### Evaluation metrics

**Mean Absolute Error (MAE)**

Measures the average absolute prediction error.

**Root Mean Squared Error (RMSE)**

Penalizes larger prediction errors more heavily.

**R² Score**

Measures the proportion of variance explained by the predictions relative to a mean-value baseline.

### Percentage improvement

Percentage improvement is calculated as:

`((Persistence Error - Hybrid Error) / Persistence Error) × 100`

Positive values indicate lower hybrid error. Negative values indicate worse hybrid performance.

RMSE and MAE are expressed in the respective pollutant measurement units. Values should not be directly compared across pollutants with different units.

## 6. Overall Hybrid Backtesting Results

| Pollutant | Selected model | Hybrid MAE | Hybrid RMSE | Hybrid R² |
|---|---|---:|---:|---:|
| PM2.5 | Hist Gradient Boosting | 18.2718 | 32.1518 | 0.7496 |
| PM10 | Persistence | 18.8163 | 31.4266 | 0.8104 |
| NO2 | Persistence | 4.4164 | 8.4015 | 0.8325 |
| SO2 | Linear Regression | 1.5813 | 2.5932 | 0.8454 |
| CO | Persistence | 0.1854 | 0.3939 | 0.8419 |
| O3 | Random Forest | 4.0263 | 6.3701 | 0.8758 |

### 6.1 Hybrid Versus Persistence

| Pollutant | MAE improvement | RMSE improvement |
|---|---:|---:|
| PM2.5 | -42.18% | -39.41% |
| PM10 | 0.00% | 0.00% |
| NO2 | 0.00% | 0.00% |
| SO2 | -20.10% | -2.04% |
| CO | 0.00% | 0.00% |
| O3 | +14.47% | +21.30% |

### 6.2 Interpretation

**PM2.5 — Hist Gradient Boosting**

The model performed worse than persistence on both MAE and RMSE. It showed substantial prediction errors, particularly in Bengaluru and Delhi.

**PM10 — Persistence**

Persistence was retained because the ML alternatives did not satisfy the validation model-selection criteria.

**NO2 — Persistence**

Persistence was retained as the selected strategy.

**SO2 — Linear Regression**

The model performed worse than persistence on both error metrics. Its overall RMSE regression was smaller than its MAE regression, but it did not demonstrate an overall improvement.

**CO — Persistence**

Persistence was retained as the selected strategy.

**O3 — Random Forest**

This was the strongest ML result in the hybrid backtest. It improved MAE by 14.47% and RMSE by 21.30% compared with persistence.

## 7. City-Wise Performance Analysis

The backtesting dataset provides observations from eight selected sensors per pollutant, one per city.

### 7.1 PM2.5

Hist Gradient Boosting improved RMSE in Hyderabad and marginally in Kolkata but underperformed persistence in the other six cities.

Notable regressions:

- Bengaluru: RMSE increased from 19.29 to 39.18.
- Ahmedabad: RMSE increased from 7.51 to 13.79.
- Delhi: RMSE increased from 36.33 to 57.14.

### 7.2 O3

Random Forest improved RMSE in six of eight cities.

Strong improvements included:

- Delhi: 55.59%.
- Kolkata: 46.10%.
- Chennai: 37.38%.
- Ahmedabad: 28.00%.

It underperformed persistence in:

- Hyderabad.
- Bengaluru.

### 7.3 SO2

Linear Regression improved RMSE in Bengaluru and Mumbai but underperformed in the other six cities.

The largest regression occurred in Hyderabad, where RMSE increased from 0.0866 to 0.3040.

### 7.4 PM10, NO2 and CO

These pollutants use persistence as their selected hybrid strategy. Their hybrid predictions therefore match the persistence baseline.

## 8. Sensor-Level Error Analysis

Sensor-level analysis was performed using the saved backtesting predictions.

### PM2.5

- Bengaluru showed significant overprediction.
- Ahmedabad also showed positive mean prediction error.
- Delhi showed negative mean prediction error.
- Hyderabad was the only sensor with a substantial MAE and RMSE improvement.
- Kolkata showed a marginal RMSE improvement.

### SO2

- Hyderabad showed substantial overprediction.
- Chennai and Kolkata showed negative mean prediction error.
- Mumbai demonstrated improvement over persistence.
- The remaining sensors generally showed worse MAE.

### O3

- Hyderabad showed significant underprediction.
- Bengaluru also showed underprediction.
- Delhi, Kolkata and Chennai demonstrated strong improvements.
- Random Forest did not perform consistently across all sensors.

### Example large-error observations

| Pollutant and city | Actual | Hybrid prediction | Persistence prediction |
|---|---:|---:|---:|
| PM2.5 Bengaluru | 193.0 | 376.73 | 217.0 |
| PM2.5 Delhi | 426.0 | 234.68 | 364.0 |
| SO2 Hyderabad | 0.925 | 1.381 | 1.030 |
| O3 Hyderabad | 43.6 | 28.85 | 43.5 |

These observations demonstrate that overall model performance can conceal important sensor-specific failures.

## 9. Prediction Engine and Quality Controls

The hybrid prediction engine implements:

- Model registry loading.
- ML pipeline loading.
- Persistence fallback.
- Historical measurement processing.
- Data freshness checks.
- Minimum 24-hour history requirements.
- Measurement coverage checks.
- Negative measurement flag checks.
- Missing observation detection.
- Prediction and rejection CSV generation.

The current historical demonstration produced:

| Metric | Result |
|---|---:|
| Sensors processed | 48 |
| Predictions generated | 3 |
| Sensors rejected | 45 |

All three generated predictions were PM2.5 predictions.

The principal rejection reasons were:

- Low-coverage observations in the 24-hour history.
- Missing hourly observations.

This conservative approach avoids generating predictions when the required historical input is incomplete.

The prediction engine currently uses historical data and must not be described as a live forecasting system.

## 10. Limitations

1. The historical collection period covers approximately seven days, which is insufficient for robust long-term forecasting claims.
2. One selected sensor per city-pollutant combination cannot represent an entire city.
3. Sensor-level test groups contain relatively few observations.
4. The current evaluation measures one-hour-ahead predictions using historical observations.
5. The test data were previously examined during exploratory model evaluation. Therefore, the backtesting results are retrospective and not an untouched independent benchmark.
6. The PM2.5 and SO2 ML models underperformed persistence in the hybrid test evaluation.
7. O3 Random Forest demonstrated improvements overall but performed poorly in Hyderabad and Bengaluru.
8. Missing measurements and varying data coverage limit prediction availability.
9. The system predicts pollutant concentrations rather than a validated AQI value.
10. The historical prediction engine does not represent real-time air-quality conditions.

## 11. Future Improvements

- Collect several months of historical measurements.
- Establish a new chronological validation period and an untouched final test period.
- Investigate city-specific and sensor-specific model errors.
- Improve training and inference data-quality consistency.
- Investigate prediction bias and extreme errors.
- Experiment with additional time-series forecasting algorithms.
- Evaluate non-negative prediction constraints without hiding model failures.
- Increase sensor coverage per city.
- Implement validated pollutant-to-AQI conversion.
- Introduce a live data collection and prediction workflow.
- Monitor prediction drift and model performance over time.
- Build an interactive dashboard for sensor-level and pollutant-level monitoring.

## 12. Conclusion

AIRSIGHT AI establishes a multi-city, multi-pollutant air-quality monitoring and next-hour prediction pipeline.

The project demonstrates the importance of comparing machine learning algorithms with a simple persistence baseline instead of assuming that more complex models always perform better.

The hybrid evaluation found that:

- O3 Random Forest provided the strongest overall ML improvement.
- PM2.5 Hist Gradient Boosting underperformed persistence.
- SO2 Linear Regression also underperformed persistence.
- Persistence remained the selected approach for PM10, NO2 and CO.

The results also demonstrate substantial differences in model performance across sensors and cities.

AIRSIGHT AI is currently a historical-data research prototype. Further data collection, independent validation, model improvement and live-data integration are required before production deployment.

The project provides a foundation for further development in air-quality analytics, time-series forecasting, model evaluation and environmental data science.