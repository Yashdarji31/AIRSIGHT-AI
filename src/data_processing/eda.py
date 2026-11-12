import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import BASE_DIR, PROCESSED_DATA_DIR

INPUT_FILE = (
    PROCESSED_DATA_DIR
    / "openaq_hourly_measurements_cleaned.csv"
)

OUTPUT_DIR = BASE_DIR / "reports" / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


def load_data():
    """Load the cleaned dataset and prepare timestamps."""

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

    df = df.dropna(
        subset=["value", "datetime_from_utc"]
    ).copy()

    # Convert UTC timestamps to Indian Standard Time.
    df["datetime_ist"] = (
        df["datetime_from_utc"]
        .dt.tz_convert("Asia/Kolkata")
    )

    df["hour_ist"] = df["datetime_ist"].dt.hour
    df["date_ist"] = df["datetime_ist"].dt.date
    df["day_name"] = df["datetime_ist"].dt.day_name()

    return df


def generate_dataset_overview(df):
    """Print and save basic dataset information."""

    print("\n========== DATASET OVERVIEW ==========")

    print("Total records:", len(df))
    print("Unique sensors:", df["sensor_id"].nunique())
    print("Cities:", df["city"].nunique())
    print("Pollutants:", sorted(df["parameter"].unique()))

    print("\nRecords by city:")
    print(df.groupby("city").size().to_string())

    print("\nRecords by pollutant:")
    print(df.groupby("parameter").size().to_string())

    overview = pd.DataFrame({
        "metric": [
            "total_records",
            "unique_sensors",
            "unique_cities",
            "unique_pollutants",
            "missing_values",
        ],
        "value": [
            len(df),
            df["sensor_id"].nunique(),
            df["city"].nunique(),
            df["parameter"].nunique(),
            int(df.isnull().sum().sum()),
        ]
    })

    overview.to_csv(
        OUTPUT_DIR / "dataset_overview.csv",
        index=False
    )


def generate_statistical_summary(df):
    """Calculate descriptive statistics."""

    summary = (
        df.groupby(["city", "parameter", "unit"])["value"]
        .agg(
            records="count",
            mean="mean",
            median="median",
            std="std",
            minimum="min",
            maximum="max"
        )
        .reset_index()
        .round(3)
    )

    summary.to_csv(
        OUTPUT_DIR / "city_pollutant_summary.csv",
        index=False
    )

    print("\n========== STATISTICAL SUMMARY ==========")
    print(summary.to_string(index=False))


def plot_city_comparisons(df):
    """Compare average pollutant concentrations across cities."""

    for pollutant in sorted(df["parameter"].unique()):

        pollutant_df = df[
            df["parameter"] == pollutant
        ]

        city_average = (
            pollutant_df.groupby("city")["value"]
            .mean()
            .reset_index()
            .sort_values("value", ascending=False)
        )

        plt.figure(figsize=(11, 6))

        sns.barplot(
            data=city_average,
            x="city",
            y="value"
        )

        unit = pollutant_df["unit"].iloc[0]

        plt.title(
            f"Average {pollutant.upper()} Concentration by City"
        )
        plt.xlabel("City")
        plt.ylabel(f"Average concentration ({unit})")
        plt.xticks(rotation=35)
        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / f"{pollutant}_city_comparison.png",
            dpi=150
        )

        plt.close()


def plot_hourly_patterns(df):
    """Explore average hourly pollutant patterns in IST."""

    for pollutant in sorted(df["parameter"].unique()):

        pollutant_df = df[
            df["parameter"] == pollutant
        ]

        hourly_average = (
            pollutant_df.groupby("hour_ist")["value"]
            .mean()
            .reset_index()
        )

        plt.figure(figsize=(11, 5))

        sns.lineplot(
            data=hourly_average,
            x="hour_ist",
            y="value",
            marker="o"
        )

        unit = pollutant_df["unit"].iloc[0]

        plt.title(
            f"Average Hourly {pollutant.upper()} Pattern (IST)"
        )
        plt.xlabel("Hour of Day (IST)")
        plt.ylabel(f"Average concentration ({unit})")
        plt.xticks(range(0, 24))
        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / f"{pollutant}_hourly_pattern.png",
            dpi=150
        )

        plt.close()


def plot_daily_patterns(df):
    """Explore daily average concentrations."""

    for pollutant in sorted(df["parameter"].unique()):

        pollutant_df = df[
            df["parameter"] == pollutant
        ]

        daily_average = (
            pollutant_df.groupby("date_ist")["value"]
            .mean()
            .reset_index()
        )

        plt.figure(figsize=(11, 5))

        sns.lineplot(
            data=daily_average,
            x="date_ist",
            y="value",
            marker="o"
        )

        unit = pollutant_df["unit"].iloc[0]

        plt.title(
            f"Daily Average {pollutant.upper()} Concentration"
        )
        plt.xlabel("Date (IST)")
        plt.ylabel(f"Average concentration ({unit})")
        plt.xticks(rotation=35)
        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / f"{pollutant}_daily_pattern.png",
            dpi=150
        )

        plt.close()


def generate_outlier_summary(df):
    """Summarize the potential outliers identified during cleaning."""

    outliers = df[
        df["outlier_flag"].astype(str).str.lower() == "true"
    ]

    summary = (
        outliers.groupby(["city", "parameter"])
        .size()
        .reset_index(name="outlier_count")
    )

    summary.to_csv(
        OUTPUT_DIR / "outlier_summary.csv",
        index=False
    )

    print("\n========== OUTLIER SUMMARY ==========")
    print("Total flagged observations:", len(outliers))
    print(summary.to_string(index=False))


def main():
    print("Starting AIRSIGHT AI exploratory data analysis...\n")

    df = load_data()

    generate_dataset_overview(df)
    generate_statistical_summary(df)
    plot_city_comparisons(df)
    plot_hourly_patterns(df)
    plot_daily_patterns(df)
    generate_outlier_summary(df)

    print("\n========== EDA COMPLETED ==========")
    print("Reports saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
