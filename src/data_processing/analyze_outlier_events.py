import pandas as pd

from src.config import PROCESSED_DATA_DIR, BASE_DIR


INPUT_FILE = (
    PROCESSED_DATA_DIR
    / "openaq_hourly_measurements_cleaned.csv"
)

OUTPUT_DIR = BASE_DIR / "reports" / "eda"

GROUP_COLUMNS = ["city", "sensor_id", "parameter"]


def main():

    print("Starting corrected outlier event analysis...\n")

    df = pd.read_csv(INPUT_FILE)

    # Normalize column names.
    df.columns = df.columns.str.strip().str.lower()

    # Support the alternative city column name.
    if "city" not in df.columns and "target_city" in df.columns:
        df = df.rename(columns={"target_city": "city"})

    # Validate required columns.
    required_columns = GROUP_COLUMNS + [
        "datetime_from_utc",
        "value",
        "outlier_flag",
        "location_name",
        "unit"
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    # Convert data types.
    df["value"] = pd.to_numeric(
        df["value"], errors="coerce"
    )

    df["datetime_from_utc"] = pd.to_datetime(
        df["datetime_from_utc"],
        utc=True,
        errors="coerce"
    )

    df["outlier_flag"] = (
        df["outlier_flag"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("true")
    )

    df = df.dropna(
        subset=GROUP_COLUMNS + [
            "datetime_from_utc",
            "value"
        ]
    )

    # Sort observations.
    df = df.sort_values(
        GROUP_COLUMNS + ["datetime_from_utc"]
    ).copy()

    # Identify previous observation within each sensor.
    grouped = df.groupby(
        GROUP_COLUMNS,
        sort=False
    )

    previous_flag = grouped["outlier_flag"].shift(
        fill_value=False
    )

    previous_time = grouped["datetime_from_utc"].shift()

    time_gap = df["datetime_from_utc"] - previous_time

    # Start a new event for each uninterrupted sequence.
    starts_event = (
        df["outlier_flag"]
        & (
            ~previous_flag
            | (time_gap > pd.Timedelta(hours=1))
        )
    )

    # Generate event IDs separately for each sensor.
    df["event_number"] = (
        starts_event.astype(int)
        .groupby(
            [df[column] for column in GROUP_COLUMNS],
            sort=False
        )
        .cumsum()
    )

    outliers = df[df["outlier_flag"]].copy()

    # Aggregate consecutive flagged observations.
    events = (
        outliers.groupby(
            GROUP_COLUMNS + ["event_number"],
            as_index=False
        )
        .agg(
            location_name=("location_name", "first"),
            unit=("unit", "first"),
            start_utc=("datetime_from_utc", "min"),
            end_utc=("datetime_from_utc", "max"),
            observations=("value", "count"),
            average_value=("value", "mean"),
            peak_value=("value", "max"),
            minimum_value=("value", "min")
        )
    )

    events["start_ist"] = (
        events["start_utc"].dt.tz_convert("Asia/Kolkata")
    )

    events["end_ist"] = (
        events["end_utc"].dt.tz_convert("Asia/Kolkata")
    )

    events["duration_hours"] = (
        events["end_utc"] - events["start_utc"]
    ).dt.total_seconds() / 3600

    events = events.sort_values(
        ["observations", "peak_value"],
        ascending=[False, False]
    )

    # Save results.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / "outlier_events.csv"

    events.to_csv(output_file, index=False)

    # Display results.
    print("========== EVENT SUMMARY ==========")

    print("Total flagged observations:", len(outliers))
    print("Total outlier events:", len(events))

    print("\n========== EVENTS BY POLLUTANT ==========")

    print(
        events.groupby("parameter")
        .size()
        .sort_values(ascending=False)
        .to_string()
    )

    print("\n========== LONGEST OUTLIER EVENTS ==========")

    columns = [
        "city",
        "location_name",
        "parameter",
        "observations",
        "average_value",
        "peak_value",
        "unit",
        "duration_hours",
        "start_ist",
        "end_ist"
    ]

    print(
        events.head(20)[columns].to_string(index=False)
    )

    print("\nReport saved:")
    print(output_file)

    print("\nOutlier event analysis completed.")


if __name__ == "__main__":
    main()
