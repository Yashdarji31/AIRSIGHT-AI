import pandas as pd

from src.config import RAW_DATA_DIR

INPUT_FILE = RAW_DATA_DIR / "openaq_hourly_measurements.csv"

POLLUTANTS = ["pm25", "pm10", "no2", "so2", "co", "o3"]

def main():
    print("Starting AIRSIGHT AI data-quality validation...\n")

    df = pd.read_csv(INPUT_FILE)

    # Convert values and timestamps to usable data types.
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["datetime_from_utc"] = pd.to_datetime(
        df["datetime_from_utc"],
        utc=True,
        errors="coerce"
    )

    print("========== BASIC DATASET INFORMATION ==========")
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

    duplicate_count = df.duplicated(
        subset=duplicate_columns,
        keep=False
    ).sum()

    print("Records involved in duplicate groups:", duplicate_count)

    print("\n========== INVALID VALUES ==========")
    print("Missing or invalid numeric values:", df["value"].isna().sum())
    print("Negative measurements:", (df["value"] < 0).sum())
    print("Invalid timestamps:", df["datetime_from_utc"].isna().sum())

    print("\n========== CITY-WISE RECORD COUNTS ==========")
    print(
        df.groupby("city").size()
        .sort_values(ascending=False)
        .to_string()
    )

    print("\n========== POLLUTANT-WISE RECORD COUNTS ==========")
    print(
        df.groupby("parameter").size()
        .sort_values(ascending=False)
        .to_string()
    )

    print("\n========== CITY AND POLLUTANT COVERAGE ==========")
    coverage = pd.crosstab(df["city"], df["parameter"])
    print(coverage.to_string())

    print("\n========== POLLUTANT UNIT COMBINATIONS ==========")
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
            "missing_hour_estimate": missing_hours,
        })

    gaps = pd.DataFrame(gap_results)

    if not gaps.empty:
        print(
            gaps.sort_values(
                "missing_hour_estimate",
                ascending=False
            ).to_string(index=False)
        )
    else:
        print("Not enough timestamp data to calculate gaps.")

    print("\n========== VALIDATION COMPLETE ==========")
    print("No data has been modified or deleted.")


if __name__ == "__main__":
    main()
