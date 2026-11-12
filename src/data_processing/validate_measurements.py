import pandas as pd

from src.config import RAW_DATA_DIR

INPUT_FILE = RAW_DATA_DIR / "openaq_hourly_measurements.csv"


def main():
    print("Starting AIRSIGHT AI data-quality validation...\n")

    df = pd.read_csv(INPUT_FILE)

    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df["datetime_from_utc"] = pd.to_datetime(
        df["datetime_from_utc"],
        utc=True,
        errors="coerce"
    )

    print("========== BASIC INFORMATION ==========")
    print("Total records:", len(df))
    print("Unique sensors:", df["sensor_id"].nunique())
    print("Cities:", df["city"].nunique())
    print("Pollutants:", sorted(df["parameter"].dropna().unique()))

    print("\n========== MISSING VALUES ==========")
    print(df.isnull().sum().to_string())

    print("\n========== DUPLICATE CHECK ==========")
    duplicate_columns = [
        "sensor_id",
        "parameter",
        "datetime_from_utc"
    ]

    print(
        "Duplicate records:",
        df.duplicated(subset=duplicate_columns).sum()
    )

    print("\n========== INVALID VALUES ==========")
    print("Invalid numeric values:", df["value"].isna().sum())
    print("Negative measurements:", (df["value"] < 0).sum())
    print("Invalid timestamps:", df["datetime_from_utc"].isna().sum())

    print("\n========== CITY-WISE RECORD COUNTS ==========")
    print(df.groupby("city").size().to_string())

    print("\n========== POLLUTANT-WISE RECORD COUNTS ==========")
    print(df.groupby("parameter").size().to_string())

    print("\n========== UNIT COMBINATIONS ==========")
    print(
        df.groupby(["parameter", "unit"], dropna=False)
        .size()
        .to_string()
    )

    print("\n========== TIMESTAMP RANGE ==========")
    print("Earliest:", df["datetime_from_utc"].min())
    print("Latest:", df["datetime_from_utc"].max())

    print("\n========== HOURLY DATA GAPS ==========")

    valid_rows = df.dropna(
        subset=["sensor_id", "datetime_from_utc"]
    )

    gap_results = []

    for sensor_id, group in valid_rows.groupby("sensor_id"):

        timestamps = (
            group["datetime_from_utc"]
            .drop_duplicates()
            .sort_values()
        )

        if len(timestamps) < 2:
            continue

        differences = timestamps.diff().dropna()

        missing_hours = sum(
            max(0, int(delta.total_seconds() // 3600) - 1)
            for delta in differences
        )

        gap_results.append({
            "sensor_id": sensor_id,
            "city": group["city"].iloc[0],
            "parameter": group["parameter"].iloc[0],
            "observed_records": len(timestamps),
            "missing_hour_estimate": missing_hours
        })

    if gap_results:
        gaps = pd.DataFrame(gap_results)

        print(
            gaps.sort_values(
                "missing_hour_estimate",
                ascending=False
            ).to_string(index=False)
        )
    else:
        print("No sufficient timestamp data for gap analysis.")

    print("\n========== VALIDATION COMPLETE ==========")


if __name__ == "__main__":
    main()
