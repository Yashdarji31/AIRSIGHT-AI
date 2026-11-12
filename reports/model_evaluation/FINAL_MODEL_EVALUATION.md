# AIRSIGHT AI — Model Evaluation Report

## 1. Project Overview

AIRSIGHT AI is an air-quality monitoring and next-hour pollutant prediction project covering eight Indian cities:

- Ahmedabad
- Bengaluru
- Chennai
- Delhi
- Hyderabad
- Kolkata
- Mumbai
- Pune

The system evaluates six pollutants: PM2.5, PM10, NO2, SO2, CO, and O3.

## 2. Dataset Overview

| Metric | Value |
|---|---:|
| Selected sensors | 48 |
| Cities | 8 |
| Pollutants | 6 |
| Feature dataset rows | 6,863 |
| Test predictions | 1,526 |
| Prediction horizon | 1 hour |

The dataset uses one selected sensor per city-pollutant combination. It does not represent a complete city-wide sensor network.

## 3. Methodology

A chronological 80/20 train-test split was used for each pollutant, with timestamps grouped to prevent observations from the same timestamp being divided between training and testing.

The following approaches were evaluated:

1. Persistence baseline
2. Linear Regression
3. Random Forest Regressor
4. HistGradientBoosting Regressor

Evaluation metrics:

- MAE: Mean Absolute Error
- RMSE: Root Mean Squared Error
- R²: Coefficient of Determination

## 4. Overall Model Performance

| Pollutant | Best Candidate | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| PM2.5 | Persistence | 12.851 | 23.062 | 0.871 |
| PM10 | Random Forest | 18.154 | 28.237 | 0.847 |
| NO2 | Linear Regression | 4.357 | 7.547 | 0.865 |
| SO2 | Persistence | 1.317 | 2.541 | 0.852 |
| CO | Random Forest | 0.187 | 0.364 | 0.865 |
| O3 | Random Forest | 4.004 | 6.322 | 0.878 |

RMSE values are expressed in their respective pollutant measurement units and should not be directly compared across pollutants.

## 5. ML Versus Persistence

| ML Model | Better Groups | Worse Groups | Total Groups |
|---|---:|---:|---:|
| Random Forest | 26 | 22 | 48 |
| Linear Regression | 24 | 24 | 48 |
| HistGradientBoosting | 20 | 28 | 48 |

Random Forest achieved the highest win rate among the evaluated ML models.

Median percentage RMSE improvements across sensor-pollutant groups:

| Model | Median Improvement |
|---|---:|
| Random Forest | +1.47% |
| Linear Regression | -0.70% |
| HistGradientBoosting | -5.41% |

Positive values indicate improvement over persistence.

## 6. Pollutant-Specific Findings

- **PM2.5:** Persistence achieved the lowest overall RMSE.
- **PM10:** Random Forest achieved the lowest overall RMSE, although performance varied across sensors.
- **NO2:** Linear Regression achieved the lowest overall RMSE.
- **SO2:** Persistence achieved the lowest overall RMSE.
- **CO:** Random Forest achieved the lowest overall RMSE.
- **O3:** Random Forest achieved the lowest overall RMSE and showed strong sensor-level improvement over persistence.

## 7. Residual Analysis

Residuals were calculated as:

Predicted value - Actual value

Notable findings:

- PM2.5 HistGradientBoosting showed positive average residual bias.
- O3 ML models showed negative average residual bias.
- The four negative predictions were produced by Linear Regression: one SO2 prediction and three O3 predictions.
- Negative concentration predictions require investigation before deployment.

## 8. Limitations

1. The measurement period covers approximately seven days, which is insufficient to establish robust long-term forecasting performance.
2. One selected sensor per city-pollutant combination cannot represent the entire city.
3. Sensor-level test groups contain relatively few observations.
4. The current evaluation measures rolling one-hour-ahead prediction using previously observed measurements.
5. Model selection was explored using the test results; an independent validation period and untouched final test period are needed for rigorous final performance claims.
6. The model predicts pollutant concentrations, not a validated AQI value.

## 9. Provisional Model Selection

| Pollutant | Selected Candidate |
|---|---|
| PM2.5 | Persistence |
| PM10 | Random Forest |
| NO2 | Linear Regression |
| SO2 | Persistence |
| CO | Random Forest |
| O3 | Random Forest |

These choices are provisional and require further validation before production use.

## 10. Future Improvements

- Collect a longer historical dataset.
- Introduce a separate validation period.
- Investigate negative predictions and outliers.
- Experiment with non-negative prediction constraints.
- Improve sensor coverage and missing-data handling.
- Evaluate additional forecasting algorithms.
- Implement validated pollutant-to-AQI conversion.
- Monitor prediction drift and model performance over time.

## 11. Conclusion

The evaluation demonstrates that model performance varies across pollutants and sensor locations. Random Forest is the strongest overall ML candidate in sensor-level baseline comparisons, while persistence remains competitive and outperforms ML overall for PM2.5 and SO2.

AIRSIGHT AI establishes an initial multi-city air-quality prediction pipeline and provides a foundation for further research, validation, and deployment.