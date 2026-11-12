import pandas as pd

from src.config import PROCESSED_DATA_DIR, BASE_DIR

INPUT_FILE = (
    PROCESSED_DATA_DIR
    / "openaq_hourly_measurements_cleaned.csv"
)

OUTPUT_DIR = BASE_DIR / "reports" / "eda"


def main():
    print("Starting AIRSIGHT AI outlier investigation...\n")

    df = pd.read_csv(INPUT_FILE)

    df["value"] = pd.to_numeric(
        df["value"],
        errors="coerce"
    )

    df["datetime_from_utc"] = pd.to_datetime(
        df["datetime_from_utc"],
        utc=True,
        errors="coerce"
    )

    df["datetime_ist"] = (
        df["datetime_from_utc"]
        .dt.tz_convert("Asia/Kolkata")
    )

    df["outlier_flag"] = (
        df["outlier_flag"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    outliers = df[df["outlier_flag"]].copy()

    print("Total flagged observations:", len(outliers))

    print("\n========== OUTLIERS BY POLLUTANT ==========")

    print(
        outliers.groupby("parameter")
        .size()
        .sort_values(ascending=False)
        .to_string()
    )

    print("\n========== TOP 10 HIGHEST OUTLIERS PER POLLUTANT ==========")

    top_outliers = (
        outliers.sort_values("value", ascending=False)
        .groupby("parameter")
        .head(10)
    )

    columns = [
        "city",
        "location_name",
        "sensor_id",
        "parameter",
        "value",
        "unit",
        "datetime_ist",
    ]

    print(top_outliers[columns].to_string(index=False))

    top_outliers[columns].to_csv(
        OUTPUT_DIR / "top_outliers.csv",
        index=False
    )

    outliers[columns].to_csv(
        OUTPUT_DIR / "all_flagged_outliers.csv",
        index=False
    )

    print("\nReports saved:")
    print(OUTPUT_DIR / "top_outliers.csv")
    print(OUTPUT_DIR / "all_flagged_outliers.csv")

    print("\nOutlier investigation completed.")


if __name__ == "__main__":
    main()
