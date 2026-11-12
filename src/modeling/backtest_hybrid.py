import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "processed" / "modeling"
MODEL_DIR = BASE_DIR / "models" / "hybrid"
REGISTRY_FILE = MODEL_DIR / "hybrid_model_registry.json"

OUTPUT_DIR = BASE_DIR / "reports" / "model_evaluation" / "hybrid_backtest"

POLLUTANTS = ["pm25", "pm10", "no2", "so2", "co", "o3"]

FEATURE_COLUMNS = [
    "city",
    "sensor_id",
    "latitude",
    "longitude",
    "lag_1h",
    "lag_3h",
    "lag_6h",
    "lag_24h",
    "rolling_mean_3h",
    "rolling_mean_6h",
    "rolling_mean_24h",
    "hour",
    "day_of_week",
    "month",
]


def load_registry():
    """Load the final hybrid model registry."""

    if not REGISTRY_FILE.exists():
        raise FileNotFoundError(
            f"Hybrid model registry not found: {REGISTRY_FILE}"
        )

    with open(REGISTRY_FILE, "r", encoding="utf-8") as file:
        registry = json.load(file)

    models = {
        item["pollutant"]: item
        for item in registry["models"]
    }

    missing = set(POLLUTANTS) - set(models)

    if missing:
        raise ValueError(
            f"Missing pollutants in model registry: {sorted(missing)}"
        )

    return models


def calculate_metrics(actual, predicted):
    """Calculate regression evaluation metrics."""

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    r2 = (
        r2_score(actual, predicted)
        if len(actual) >= 2
        else np.nan
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


def calculate_improvement(baseline, hybrid):
    """Calculate percentage error reduction over persistence."""

    if baseline == 0:
        return np.nan

    return ((baseline - hybrid) / baseline) * 100


def load_hybrid_model(pollutant, registry_entry):
    """Load the selected ML pipeline when required."""

    selected_model = registry_entry["selected_model"]

    if selected_model == "persistence_baseline":
        return None

    filename = Path(registry_entry["model_path"]).name
    model_path = MODEL_DIR / filename

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {model_path}"
        )

    print(f"Loading {pollutant.upper()}: {selected_model}")

    return joblib.load(model_path)


def evaluate_pollutant(pollutant, registry_entry):
    """Backtest one pollutant using its selected hybrid model."""

    test_file = DATA_DIR / pollutant / "test.csv"

    if not test_file.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {test_file}"
        )

    df = pd.read_csv(test_file)

    required_columns = (
        FEATURE_COLUMNS
        + ["target_value", "target_datetime_utc"]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{pollutant}: missing columns {missing_columns}"
        )

    # Ensure target and persistence inputs are numeric.
    df["target_value"] = pd.to_numeric(
        df["target_value"], errors="coerce"
    )

    df["lag_1h"] = pd.to_numeric(
        df["lag_1h"], errors="coerce"
    )

    # Only retain rows where both approaches can be evaluated.
    valid_mask = (
        df["target_value"].notna()
        & df["lag_1h"].notna()
        & df["target_datetime_utc"].notna()
    )

    df = df.loc[valid_mask].copy()

    if df.empty:
        raise ValueError(
            f"No valid evaluation rows for {pollutant}."
        )

    actual = df["target_value"].to_numpy(dtype=float)

    # Persistence predicts that the next value equals lag_1h.
    persistence_predictions = df["lag_1h"].to_numpy(
        dtype=float
    )

    selected_model = registry_entry["selected_model"]

    model = load_hybrid_model(pollutant, registry_entry)

    if model is None:
        hybrid_predictions = persistence_predictions.copy()
    else:
        X = df[FEATURE_COLUMNS].copy()

        hybrid_predictions = model.predict(X)

    # Evaluate both approaches on identical observations.
    hybrid_metrics = calculate_metrics(
        actual,
        hybrid_predictions,
    )

    persistence_metrics = calculate_metrics(
        actual,
        persistence_predictions,
    )

    prediction_results = pd.DataFrame({
        "city": df["city"].to_numpy(),
        "sensor_id": df["sensor_id"].to_numpy(),
        "pollutant": pollutant,
        "target_datetime_utc": df[
            "target_datetime_utc"
        ].to_numpy(),
        "actual_value": actual,
        "hybrid_prediction": hybrid_predictions,
        "persistence_prediction": persistence_predictions,
        "selected_model": selected_model,
    })

    prediction_results["hybrid_absolute_error"] = (
        np.abs(
            prediction_results["actual_value"]
            - prediction_results["hybrid_prediction"]
        )
    )

    prediction_results["persistence_absolute_error"] = (
        np.abs(
            prediction_results["actual_value"]
            - prediction_results["persistence_prediction"]
        )
    )

    metric_result = {
        "pollutant": pollutant,
        "selected_model": selected_model,
        "evaluation_rows": len(df),

        "hybrid_MAE": hybrid_metrics["MAE"],
        "hybrid_RMSE": hybrid_metrics["RMSE"],
        "hybrid_R2": hybrid_metrics["R2"],

        "persistence_MAE": persistence_metrics["MAE"],
        "persistence_RMSE": persistence_metrics["RMSE"],
        "persistence_R2": persistence_metrics["R2"],

        "MAE_improvement_percent": calculate_improvement(
            persistence_metrics["MAE"],
            hybrid_metrics["MAE"],
        ),

        "RMSE_improvement_percent": calculate_improvement(
            persistence_metrics["RMSE"],
            hybrid_metrics["RMSE"],
        ),
    }

    # City-wise evaluation.
    city_results = []

    for city, city_df in prediction_results.groupby("city"):
        city_actual = city_df["actual_value"]
        city_hybrid = city_df["hybrid_prediction"]
        city_persistence = city_df["persistence_prediction"]

        city_hybrid_metrics = calculate_metrics(
            city_actual,
            city_hybrid,
        )

        city_persistence_metrics = calculate_metrics(
            city_actual,
            city_persistence,
        )

        city_results.append({
            "pollutant": pollutant,
            "city": city,
            "selected_model": selected_model,
            "evaluation_rows": len(city_df),

            "hybrid_MAE": city_hybrid_metrics["MAE"],
            "hybrid_RMSE": city_hybrid_metrics["RMSE"],
            "hybrid_R2": city_hybrid_metrics["R2"],

            "persistence_MAE": city_persistence_metrics["MAE"],
            "persistence_RMSE": city_persistence_metrics["RMSE"],
            "persistence_R2": city_persistence_metrics["R2"],
        })

    print(
        f"{pollutant.upper()}: "
        f"{len(df)} rows evaluated successfully."
    )

    return prediction_results, metric_result, city_results


def main():
    print("\nAIRSIGHT AI — HYBRID MODEL BACKTEST")
    print("=" * 65)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    registry = load_registry()

    all_predictions = []
    all_metrics = []
    all_city_metrics = []

    for pollutant in POLLUTANTS:
        print(f"\nEvaluating {pollutant.upper()}...")

        predictions, metrics, city_metrics = (
            evaluate_pollutant(
                pollutant,
                registry[pollutant],
            )
        )

        all_predictions.append(predictions)
        all_metrics.append(metrics)
        all_city_metrics.extend(city_metrics)

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    metrics_df = pd.DataFrame(all_metrics)

    city_metrics_df = pd.DataFrame(all_city_metrics)

    predictions_path = (
        OUTPUT_DIR / "hybrid_backtest_predictions.csv"
    )

    metrics_path = (
        OUTPUT_DIR / "hybrid_backtest_metrics.csv"
    )

    city_metrics_path = (
        OUTPUT_DIR / "hybrid_backtest_city_metrics.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    city_metrics_df.to_csv(
        city_metrics_path,
        index=False,
    )

    print("\n" + "=" * 65)
    print("HYBRID BACKTEST SUMMARY")
    print("=" * 65)

    print("Total evaluation rows:", len(predictions_df))

    print("\nPollutant-wise metrics:")

    display_columns = [
        "pollutant",
        "selected_model",
        "evaluation_rows",
        "hybrid_MAE",
        "hybrid_RMSE",
        "persistence_MAE",
        "persistence_RMSE",
        "MAE_improvement_percent",
        "RMSE_improvement_percent",
    ]

    print(
        metrics_df[display_columns].round(4).to_string(
            index=False
        )
    )

    print("\nSaved reports:")
    print(predictions_path)
    print(metrics_path)
    print(city_metrics_path)

    print("\nHybrid backtesting completed successfully.")


if __name__ == "__main__":
    main()
