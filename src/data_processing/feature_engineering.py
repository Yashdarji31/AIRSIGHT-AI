import pandas as pd

from src.config import PROCESSED_DATA_DIR, BASE_DIR


INPUT_FILE = (
    PROCESSED_DATA_DIR
    / "openaq_hourly_measurements_cleaned.csv"
)

OUTPUT_FILE = PROCESSED_DATA_DIR / "openaq_features.csv"

REPORT_DIR = BASE_DIR / "reports" / "eda"

GROUP_COLUMNS = ["city", "sensor_id", "parameter"]


def create_features(group):

    group = group.sort_values("datetime_from_utc").copy()

    metadata_columns = [
        "city", "sensor_id", "parameter", "unit",
        "location_name", "latitude", "longitude"
    ]

    metadata = {
        col: group[col].iloc[0]
        for col in metadata_columns
        if col in group.columns
    }

    values = (
        group.set_index("datetime_from_utc")["value"]
        .sort_index()
    )

    if values.index.has_duplicates:
        raise ValueError("Duplicate timestamps found.")

    hourly_index = pd.date_range(
        start=values.index.min(),
        end=values.index.max(),
        freq="h",
        tz="UTC"
    )

    values = values.reindex(hourly_index)

    features = pd.DataFrame(index=hourly_index)

    features["lag_1h"] = values
    features["lag_3h"] = values.shift(2)
    features["lag_6h"] = values.shift(5)
    features["lag_24h"] = values.shift(23)

    features["rolling_mean_3h"] = (
        values.rolling(3, min_periods=3).mean()
    )

    features["rolling_mean_6h"] = (
        values.rolling(6, min_periods=6).mean()
    )

    features["rolling_mean_24h"] = (
        values.rolling(24, min_periods=24).mean()
    )

    features["target_value"] = values.shift(-1)

    features["target_datetime_utc"] = (
        features.index + pd.Timedelta(hours=1)
    )

    # Keep only real consecutive hourly observation pairs.
    valid_pairs = (
        values.notna()
        & features["target_value"].notna()
    )

    features = features.loc[valid_pairs].copy()

    # Restore metadata after reindexing.
    for column, value in metadata.items():
        features[column] = value

    prediction_time_ist = (
        features["target_datetime_utc"]
        .dt.tz_convert("Asia/Kolkata")
    )

    features["hour"] = prediction_time_ist.dt.hour

    features["day_of_week"] = (
        prediction_time_ist.dt.day_of_week
    )

    features["month"] = prediction_time_ist.dt.month

    return features.reset_index(
        names="datetime_from_utc"
    )


def main():

    print("Starting corrected AIRSIGHT AI feature engineering...\n")

    df = pd.read_csv(INPUT_FILE)

    df.columns = df.columns.str.strip().str.lower()

    required = GROUP_COLUMNS + [
        "datetime_from_utc", "value", "unit"
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["datetime_from_utc"] = pd.to_datetime(
        df["datetime_from_utc"],
        utc=True,
        errors="coerce"
    )

    df["value"] = pd.to_numeric(
        df["value"],
        errors="coerce"
    )

    df = df.dropna(
        subset=required
    )

    feature_frames = []

    for _, group in df.groupby(GROUP_COLUMNS, sort=False):

        features = create_features(group)

        if not features.empty:
            feature_frames.append(features)

    if not feature_frames:
        raise ValueError("No valid feature rows generated.")

    result = pd.concat(feature_frames, ignore_index=True)

    feature_columns = [
        "datetime_from_utc",
        "target_datetime_utc",
        "city",
        "sensor_id",
        "parameter",
        "unit",
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
        "target_value"
    ]

    feature_columns += [
        col for col in [
            "location_name", "latitude", "longitude"
        ]
        if col in result.columns
    ]

    result = result[feature_columns].copy()

    # Validate metadata.
    metadata_columns = [
        "city", "sensor_id", "parameter", "unit"
    ]

    if result[metadata_columns].isna().any().any():
        raise ValueError("Missing metadata detected.")

    # Validate next-hour target alignment.
    time_difference = (
        result["target_datetime_utc"]
        - result["datetime_from_utc"]
    )

    if not time_difference.eq(
        pd.Timedelta(hours=1)
    ).all():
        raise ValueError("Invalid target timestamp alignment.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    result.to_csv(OUTPUT_FILE, index=False)

    quality_report = pd.DataFrame({
        "column": result.columns,
        "missing_values": result.isna().sum().values,
        "missing_percentage": (
            result.isna().mean().values * 100
        )
    })

    quality_report.to_csv(
        REPORT_DIR / "feature_quality_report.csv",
        index=False
    )

    print("========== FEATURE ENGINEERING SUMMARY ==========")

    print("Original observations:", len(df))
    print("Generated feature rows:", len(result))
    print("Unique cities:", result["city"].nunique())
    print("Unique sensors:", result["sensor_id"].nunique())
    print("Pollutants:", result["parameter"].nunique())

    print("\n========== MISSING FEATURE VALUES ==========")

    print(result.isna().sum().to_string())

    print("\n========== SAMPLE FEATURE DATA ==========")

    print(result.head(10).to_string(index=False))

    print("\nSaved feature dataset:", OUTPUT_FILE)
    print("Saved quality report:", REPORT_DIR / "feature_quality_report.csv")

    print("\nCorrected feature engineering completed successfully.")


if __name__ == "__main__":
    main()
