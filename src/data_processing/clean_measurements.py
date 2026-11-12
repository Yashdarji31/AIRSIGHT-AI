import pandas as pd

from src.config import PROCESSED_DATA_DIR

INPUT_FILE = (
    PROCESSED_DATA_DIR
    / "openaq_hourly_measurements_enriched.csv"
)

OUTPUT_FILE = (
    PROCESSED_DATA_DIR
    / "openaq_hourly_measurements_cleaned.csv"
)


def main():
    print("Starting AIRSIGHT AI data cleaning...\n")

    # Load enriched dataset
    df = pd.read_csv(INPUT_FILE)

    original_count = len(df)

    # Convert numeric values safely
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Parse timestamps as UTC
    df["datetime_from_utc"] = pd.to_datetime(
        df["datetime_from_utc"],
        utc=True,
        errors="coerce"
    )

    df["datetime_to_utc"] = pd.to_datetime(
        df["datetime_to_utc"],
        utc=True,
        errors="coerce"
    )

    # Required fields for a usable measurement
    required_columns = [
        "sensor_id",
        "city",
        "parameter",
        "unit",
        "value",
        "datetime_from_utc",
        "datetime_to_utc",
        "latitude",
        "longitude",
    ]

    print("========== INITIAL QUALITY CHECK ==========")
    print("Original records:", original_count)
    print("Missing required fields:")
    print(df[required_columns].isna().sum().to_string())

    # Remove records only when essential measurement information is invalid
    before_removal = len(df)

    df = df.dropna(subset=required_columns).copy()

    removed_invalid = before_removal - len(df)

    # Remove exact duplicate sensor-time measurements
    before_duplicates = len(df)

    df = df.drop_duplicates(
        subset=[
            "sensor_id",
            "parameter",
            "datetime_from_utc",
        ],
        keep="first"
    ).copy()

    removed_duplicates = before_duplicates - len(df)

    # Flag negative values rather than silently deleting them
    df["negative_value_flag"] = df["value"] < 0

    # Flag unusually high values using the IQR method.
    # This is an investigation flag, NOT an automatic deletion.
    # Detect unusual readings separately for each sensor
    df["outlier_flag"] = False

    for (sensor_id, parameter, unit), group in df.groupby(
        ["sensor_id", "parameter", "unit"]
    ):
        q1 = group["value"].quantile(0.25)
        q3 = group["value"].quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        mask = (
            (df["sensor_id"] == sensor_id)
            & (df["parameter"] == parameter)
            & (df["unit"] == unit)
            & (
                (df["value"] < lower_bound)
                | (df["value"] > upper_bound)
            )
        )

        df.loc[mask, "outlier_flag"] = True
    # Sort records chronologically
    df = df.sort_values(
        ["city", "sensor_id", "datetime_from_utc"]
    ).reset_index(drop=True)

    # Add useful time features in UTC
    df["hour"] = df["datetime_from_utc"].dt.hour
    df["day"] = df["datetime_from_utc"].dt.day
    df["day_of_week"] = df["datetime_from_utc"].dt.day_of_week
    df["month"] = df["datetime_from_utc"].dt.month

    # Save cleaned dataset
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n========== CLEANING SUMMARY ==========")
    print("Original records:", original_count)
    print("Invalid records removed:", removed_invalid)
    print("Duplicate records removed:", removed_duplicates)
    print("Final records:", len(df))
    print("Flagged negative values:", df["negative_value_flag"].sum())
    print("Flagged potential outliers:", df["outlier_flag"].sum())

    print("\n========== FINAL DATASET CHECK ==========")
    print("Missing values:")
    print(df[required_columns].isna().sum().to_string())

    print("\nRecords by city:")
    print(df.groupby("city").size().to_string())

    print("\nRecords by pollutant:")
    print(df.groupby("parameter").size().to_string())

    print("\nOutput file:", OUTPUT_FILE)
    print("\nData cleaning completed.")


if __name__ == "__main__":
    main()
