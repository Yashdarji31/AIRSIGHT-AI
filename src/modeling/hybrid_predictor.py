import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ==================================================
# 1. CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "openaq_hourly_measurements_cleaned.csv"
)

REGISTRY_FILE = (
    BASE_DIR
    / "models"
    / "hybrid"
    / "hybrid_model_registry.json"
)

OUTPUT_DIR = (
    BASE_DIR
    / "reports"
    / "predictions"
)

PREDICTION_FILE = OUTPUT_DIR / "hybrid_predictions.csv"
REJECTED_FILE = OUTPUT_DIR / "rejected_sensors.csv"

POLLUTANTS = ["pm25", "pm10", "no2", "so2", "co", "o3"]

NUMERIC_FEATURES = [
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

CATEGORICAL_FEATURES = ["city", "sensor_id"]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Historical demonstration only: the current source data is old.
# Live predictions require fresh observations.
MAX_DATA_AGE_HOURS = 2

# Conservative quality threshold for inference.
MIN_COVERAGE_PERCENT = 75.0

MIN_HISTORY_HOURS = 24


# ==================================================
# 2. LOAD REGISTRY AND MODELS
# ==================================================

def load_registry():

    if not REGISTRY_FILE.exists():
        raise FileNotFoundError(REGISTRY_FILE)

    with open(REGISTRY_FILE, "r", encoding="utf-8") as file:
        registry = json.load(file)

    if registry.get("architecture") != "hybrid":
        raise ValueError("Invalid model registry architecture.")

    models = registry.get("models", [])

    model_map = {
        item["pollutant"]: item
        for item in models
    }

    if set(model_map) != set(POLLUTANTS):
        raise ValueError(
            "Registry must contain exactly six pollutant models."
        )

    return model_map


def load_ml_models(model_map):

    loaded_models = {}

    for pollutant, details in model_map.items():

        model_name = details["selected_model"]

        if model_name == "persistence_baseline":
            continue

        if details.get("status") != "trained_successfully":
            raise ValueError(
                f"{pollutant}: ML model is not marked as trained."
            )

        model_path = Path(details["model_path"])

        if not model_path.is_absolute():
            model_path = BASE_DIR / model_path

        if not model_path.exists():
            raise FileNotFoundError(model_path)

        loaded_models[pollutant] = joblib.load(model_path)

    return loaded_models


# ==================================================
# 3. LOAD AND VALIDATE MEASUREMENTS
# ==================================================

def load_measurements():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    df.columns = df.columns.str.strip().str.lower()

    required = [
        "sensor_id",
        "location_name",
        "city",
        "parameter",
        "unit",
        "value",
        "datetime_from_utc",
        "latitude",
        "longitude",
        "coverage_percent",
        "negative_value_flag",
        "outlier_flag",
    ]

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Measurement dataset missing columns: {missing}"
        )

    df["datetime_from_utc"] = pd.to_datetime(
        df["datetime_from_utc"],
        utc=True,
        errors="coerce",
    )

    for column in [
        "value",
        "coverage_percent",
        "latitude",
        "longitude",
        "sensor_id",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df["parameter"] = df["parameter"].astype(str).str.lower()
    df["city"] = df["city"].astype(str).str.strip()
    df["unit"] = df["unit"].astype(str).str.strip()

    df["negative_value_flag"] = (
        df["negative_value_flag"].astype(str).str.lower()
    )

    df["outlier_flag"] = (
        df["outlier_flag"].astype(str).str.lower()
    )

    df = df.dropna(
        subset=[
            "sensor_id",
            "city",
            "parameter",
            "unit",
            "value",
            "datetime_from_utc",
            "latitude",
            "longitude",
            "coverage_percent",
        ]
    ).copy()

    df = df[df["parameter"].isin(POLLUTANTS)].copy()

    if df.duplicated(
        subset=["sensor_id", "parameter", "datetime_from_utc"]
    ).any():

        raise ValueError(
            "Duplicate sensor-pollutant timestamps detected."
        )

    return df


# ==================================================
# 4. BUILD FEATURES FOR ONE SENSOR
# ==================================================


def build_sensor_features(group):

    group = group.sort_values("datetime_from_utc").copy()

    if group["datetime_from_utc"].duplicated().any():
        return None, "Duplicate timestamps."

    latest = group.iloc[-1]
    latest_time = latest["datetime_from_utc"]

    # Check the latest measurement's quality separately.
    if latest["negative_value_flag"] != "false":
        return None, "Latest measurement has a negative-value flag."

    if latest["coverage_percent"] < MIN_COVERAGE_PERCENT:
        return None, "Latest measurement has insufficient coverage."

    reference_time = group.attrs.get("reference_time", latest_time)

    age_hours = (
        reference_time - latest_time
    ).total_seconds() / 3600

    if age_hours < 0:
        return None, "Latest observation is in the future."

    if age_hours > MAX_DATA_AGE_HOURS:
        return None, f"Stale measurement: {age_hours:.1f} hours old."

    # Preserve the original observation timeline.
    values = group.set_index("datetime_from_utc")["value"].sort_index()

    expected_index = pd.date_range(
        end=latest_time,
        periods=MIN_HISTORY_HOURS,
        freq="h",
    )

    # Reindex the original data so missing hours remain visible.
    history = values.reindex(expected_index)

    if history.isna().any():
        missing_count = int(history.isna().sum())
        return None, f"Missing hourly observations: {missing_count}."

    # Check quality for every observation in the same window.
    quality = (
        group.set_index("datetime_from_utc")
        .reindex(expected_index)
    )

    if quality["coverage_percent"].isna().any():
        return None, "Missing coverage information in history."

    if (quality["coverage_percent"] < MIN_COVERAGE_PERCENT).any():
        return None, "Low-coverage observation in 24-hour history."

    if (quality["negative_value_flag"] != "false").any():
        return None, "Negative-value flag in 24-hour history."

    if len(history) != MIN_HISTORY_HOURS:
        return None, "Insufficient historical readings."

    prediction_time_utc = latest_time + pd.Timedelta(hours=1)

    prediction_time_ist = prediction_time_utc.tz_convert(
        "Asia/Kolkata"
    )

    row = {
        "city": latest["city"],
        "sensor_id": int(latest["sensor_id"]),
        "latitude": latest["latitude"],
        "longitude": latest["longitude"],
        "lag_1h": history.iloc[-1],
        "lag_3h": history.iloc[-3],
        "lag_6h": history.iloc[-6],
        "lag_24h": history.iloc[-24],
        "rolling_mean_3h": history.iloc[-3:].mean(),
        "rolling_mean_6h": history.iloc[-6:].mean(),
        "rolling_mean_24h": history.mean(),
        "hour": prediction_time_ist.hour,
        "day_of_week": prediction_time_ist.day_of_week,
        "month": prediction_time_ist.month,
    }

    feature_df = pd.DataFrame(
        [row],
        columns=FEATURE_COLUMNS,
    )

    return {
        "features": feature_df,
        "latest": latest,
        "prediction_time_utc": prediction_time_utc,
        "age_hours": age_hours,
    }, None


    # Check freshness against the latest timestamp in the entire
    # dataset. This permits reproducible historical demonstrations.
    reference_time = group.attrs.get("reference_time")

    if reference_time is None:
        reference_time = latest_time

    age_hours = (
        reference_time - latest_time
    ).total_seconds() / 3600

    if age_hours < 0:
        return None, "Latest sensor observation is in the future."

    if age_hours > MAX_DATA_AGE_HOURS:
        return None, f"Stale measurement: {age_hours:.1f} hours old."

    values = (
        group.set_index("datetime_from_utc")["value"]
        .sort_index()
    )

    if values.index.has_duplicates:
        return None, "Duplicate timestamps."

    # Require a continuous sequence of exactly hourly readings.
    expected_index = pd.date_range(
        end=latest_time,
        periods=MIN_HISTORY_HOURS,
        freq="h",
    )

    history = values.reindex(expected_index)

    if history.isna().any():
        return None, "Insufficient consecutive 24-hour history."

    if len(history) != MIN_HISTORY_HOURS:
        return None, "Insufficient historical readings."

    # Features are calculated using observations up to latest_time.
    prediction_time_utc = latest_time + pd.Timedelta(hours=1)

    prediction_time_ist = prediction_time_utc.tz_convert(
        "Asia/Kolkata"
    )

    row = {
        "city": latest["city"],
        "sensor_id": int(latest["sensor_id"]),
        "latitude": latest["latitude"],
        "longitude": latest["longitude"],
        "lag_1h": history.iloc[-1],
        "lag_3h": history.iloc[-3],
        "lag_6h": history.iloc[-6],
        "lag_24h": history.iloc[-24],
        "rolling_mean_3h": history.iloc[-3:].mean(),
        "rolling_mean_6h": history.iloc[-6:].mean(),
        "rolling_mean_24h": history.mean(),
        "hour": prediction_time_ist.hour,
        "day_of_week": prediction_time_ist.day_of_week,
        "month": prediction_time_ist.month,
    }

    feature_df = pd.DataFrame([row], columns=FEATURE_COLUMNS)

    return {
        "features": feature_df,
        "latest": latest,
        "prediction_time_utc": prediction_time_utc,
        "age_hours": age_hours,
    }, None


# ==================================================
# 5. GENERATE HYBRID PREDICTIONS
# ==================================================

def generate_predictions(
    df,
    model_map,
    loaded_models,
    reference_time,
):

    predictions = []
    rejected = []

    grouped = df.groupby(
        ["city", "sensor_id", "parameter", "unit"],
        sort=False,
    )

    for (city, sensor_id, pollutant, unit), group in grouped:

        if pollutant not in POLLUTANTS:
            continue

        group = group.copy()

        group.attrs["reference_time"] = reference_time

        result, reason = build_sensor_features(group)

        if reason:

            rejected.append({
                "city": city,
                "sensor_id": sensor_id,
                "pollutant": pollutant,
                "reason": reason,
            })

            continue

        details = model_map[pollutant]

        selected_model = details["selected_model"]

        features = result["features"]

        if selected_model == "persistence_baseline":

            prediction = float(
                features.iloc[0]["lag_1h"]
            )

        else:

            pipeline = loaded_models[pollutant]

            prediction = float(
                pipeline.predict(features)[0]
            )

        if not np.isfinite(prediction):

            rejected.append({
                "city": city,
                "sensor_id": sensor_id,
                "pollutant": pollutant,
                "reason": "Model produced a non-finite prediction.",
            })

            continue

        warning = ""

        if prediction < 0:

            warning = (
                "Negative prediction: requires investigation."
            )

            warnings.warn(
                f"{pollutant.upper()} prediction is negative "
                f"for sensor {sensor_id}: {prediction}"
            )

        latest = result["latest"]

        predictions.append({

            "city": city,
            "sensor_id": int(sensor_id),
            "pollutant": pollutant,
            "unit": unit,

            "last_observation_utc": (
                latest["datetime_from_utc"].isoformat()
            ),

            "prediction_datetime_utc": (
                result["prediction_time_utc"].isoformat()
            ),

            "latest_observed_value": float(
                latest["value"]
            ),

            "predicted_value": prediction,

            "selected_model": selected_model,

            "data_age_hours": result["age_hours"],

            "latest_coverage_percent": float(
                latest["coverage_percent"]
            ),

            "latest_outlier_flag": (
                latest["outlier_flag"]
            ),

            "prediction_warning": warning,

            "data_status": "historical_demo",

        })

    return pd.DataFrame(predictions), pd.DataFrame(rejected)


# ==================================================
# 6. MAIN
# ==================================================

def main():

    print("\nAIRSIGHT AI — HYBRID PREDICTION ENGINE")
    print("=" * 65)

    model_map = load_registry()

    print("Model registry loaded.")

    loaded_models = load_ml_models(model_map)

    print("ML pipelines loaded:", len(loaded_models))

    df = load_measurements()

    reference_time = df["datetime_from_utc"].max()

    print("Measurements loaded:", len(df))
    print("Latest dataset timestamp:", reference_time)

    print(
        "\nIMPORTANT: This run uses historical data. "
        "It is not a live forecast."
    )

    predictions, rejected = generate_predictions(
        df,
        model_map,
        loaded_models,
        reference_time,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        PREDICTION_FILE,
        index=False,
    )

    rejected.to_csv(
        REJECTED_FILE,
        index=False,
    )

    print("\n" + "=" * 65)
    print("PREDICTION SUMMARY")
    print("=" * 65)

    print("Predictions generated:", len(predictions))
    print("Sensors rejected:", len(rejected))

    if not predictions.empty:

        print("\nPredictions by pollutant:")

        print(
            predictions.groupby("pollutant").size().to_string()
        )

        print("\nPrediction sample:")

        print(
            predictions.head(10).to_string(index=False)
        )

    if not rejected.empty:

        print("\nRejected sensor reasons:")

        print(
            rejected["reason"].value_counts().to_string()
        )

    print("\nSaved predictions:", PREDICTION_FILE)
    print("Saved rejected sensors:", REJECTED_FILE)

    print("\nHybrid prediction engine completed.")


if __name__ == "__main__":
    main()
