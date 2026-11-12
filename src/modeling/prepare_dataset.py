import pandas as pd
from pathlib import Path

from src.config import PROCESSED_DATA_DIR, BASE_DIR


# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

INPUT_FILE = PROCESSED_DATA_DIR / "openaq_features.csv"

OUTPUT_DIR = PROCESSED_DATA_DIR / "modeling"

REPORT_DIR = BASE_DIR / "reports" / "model_evaluation"

TRAIN_RATIO = 0.80

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
    "month"
]

TARGET_COLUMN = "target_value"

TIME_COLUMN = "target_datetime_utc"


# --------------------------------------------------
# DATA VALIDATION
# --------------------------------------------------

def validate_dataset(df):

    required_columns = (
        FEATURE_COLUMNS
        + [TARGET_COLUMN, TIME_COLUMN, "parameter"]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df[TARGET_COLUMN].isna().any():
        raise ValueError("Missing target values detected.")

    if df.duplicated(
        ["sensor_id", "datetime_from_utc", "parameter"]
    ).any():
        raise ValueError("Duplicate sensor timestamps detected.")

    if not set(df["parameter"].unique()).issubset(POLLUTANTS):
        raise ValueError("Unexpected pollutant found.")

    print("Dataset validation passed.")


# --------------------------------------------------
# CHRONOLOGICAL TRAIN/TEST SPLIT
# --------------------------------------------------

def split_by_time(group):

    group = group.sort_values(TIME_COLUMN).copy()

    unique_times = (
        group[TIME_COLUMN]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    split_index = int(len(unique_times) * TRAIN_RATIO)

    if split_index <= 0 or split_index >= len(unique_times):
        raise ValueError("Insufficient timestamps for splitting.")

    cutoff_time = unique_times.iloc[split_index]

    train_data = group[
        group[TIME_COLUMN] < cutoff_time
    ].copy()

    test_data = group[
        group[TIME_COLUMN] >= cutoff_time
    ].copy()

    if train_data.empty or test_data.empty:
        raise ValueError("Training or testing partition is empty.")

    if train_data[TIME_COLUMN].max() >= test_data[TIME_COLUMN].min():
        raise ValueError("Temporal leakage detected.")

    return train_data, test_data, cutoff_time


# --------------------------------------------------
# PREPARE ONE POLLUTANT
# --------------------------------------------------

def prepare_pollutant(df, pollutant):

    print(f"\nPreparing {pollutant.upper()}...")

    group = df[
        df["parameter"] == pollutant
    ].copy()

    if group.empty:
        raise ValueError(f"No observations found for {pollutant}.")

    train_data, test_data, cutoff_time = split_by_time(group)

    output_columns = FEATURE_COLUMNS + [
        TARGET_COLUMN,
        TIME_COLUMN
    ]

    train_data = train_data[output_columns].copy()
    test_data = test_data[output_columns].copy()

    # Treat sensor IDs as categorical identifiers.
    for data in [train_data, test_data]:
        data["sensor_id"] = data["sensor_id"].astype(str)
        data["city"] = data["city"].astype(str)

    pollutant_dir = OUTPUT_DIR / pollutant
    pollutant_dir.mkdir(parents=True, exist_ok=True)

    train_file = pollutant_dir / "train.csv"
    test_file = pollutant_dir / "test.csv"

    train_data.to_csv(train_file, index=False)
    test_data.to_csv(test_file, index=False)

    print("Total observations:", len(group))
    print("Training observations:", len(train_data))
    print("Testing observations:", len(test_data))
    print("Training ends:", train_data[TIME_COLUMN].max())
    print("Testing starts:", test_data[TIME_COLUMN].min())
    print("Split cutoff:", cutoff_time)

    return {
        "pollutant": pollutant,
        "total_rows": len(group),
        "train_rows": len(train_data),
        "test_rows": len(test_data),
        "unique_sensors": group["sensor_id"].nunique(),
        "unique_cities": group["city"].nunique(),
        "cutoff_time": cutoff_time,
        "train_end": train_data[TIME_COLUMN].max(),
        "test_start": test_data[TIME_COLUMN].min(),
        "train_file": str(train_file),
        "test_file": str(test_file)
    }


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("Starting AIRSIGHT AI dataset preparation...\n")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    df.columns = df.columns.str.strip().str.lower()

    df["datetime_from_utc"] = pd.to_datetime(
        df["datetime_from_utc"],
        utc=True,
        errors="coerce"
    )

    df[TIME_COLUMN] = pd.to_datetime(
        df[TIME_COLUMN],
        utc=True,
        errors="coerce"
    )

    if df[["datetime_from_utc", TIME_COLUMN]].isna().any().any():
        raise ValueError("Invalid timestamps detected.")

    validate_dataset(df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    summaries = []

    for pollutant in POLLUTANTS:

        summary = prepare_pollutant(df, pollutant)

        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)

    report_file = REPORT_DIR / "dataset_split_summary.csv"

    summary_df.to_csv(report_file, index=False)

    print("\n========== DATASET PREPARATION SUMMARY ==========")

    print(
        summary_df[
            [
                "pollutant",
                "total_rows",
                "train_rows",
                "test_rows",
                "unique_sensors",
                "unique_cities"
            ]
        ].to_string(index=False)
    )

    print("\nSaved split report:", report_file)

    print("\nDataset preparation completed successfully!")


if __name__ == "__main__":
    main()
