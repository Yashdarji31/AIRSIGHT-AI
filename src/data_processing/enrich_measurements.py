import pandas as pd

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

MEASUREMENTS_FILE = RAW_DATA_DIR / "openaq_hourly_measurements.csv"
SENSORS_FILE = RAW_DATA_DIR / "openaq_sensors.csv"

OUTPUT_FILE = PROCESSED_DATA_DIR / "openaq_hourly_measurements_enriched.csv"


def main():
    print("Starting AIRSIGHT AI metadata enrichment...\n")

    # Load both datasets
    measurements = pd.read_csv(MEASUREMENTS_FILE)
    sensors = pd.read_csv(SENSORS_FILE)

    # Keep only the metadata required for enrichment
    metadata_columns = [
        "sensor_id",
        "location_id",
        "latitude",
        "longitude",
    ]

    sensor_metadata = sensors[metadata_columns].copy()

    # Normalize sensor IDs before joining
    measurements["sensor_id"] = measurements["sensor_id"].astype(str)
    sensor_metadata["sensor_id"] = sensor_metadata["sensor_id"].astype(str)

    # Each sensor should map to one unique metadata record
    sensor_metadata = sensor_metadata.drop_duplicates(
        subset=["sensor_id"]
    )

    # Join sensor metadata to hourly measurements
    enriched = measurements.drop(
        columns=["latitude", "longitude"],
        errors="ignore"
    ).merge(
        sensor_metadata,
        on="sensor_id",
        how="left",
        validate="many_to_one"
    )

    # Save the enriched dataset
    enriched.to_csv(OUTPUT_FILE, index=False)

    print("========== ENRICHMENT SUMMARY ==========")
    print("Original measurement records:", len(measurements))
    print("Enriched measurement records:", len(enriched))
    print("Unique sensors:", enriched["sensor_id"].nunique())

    print("\nMissing metadata:")
    print(
        enriched[
            ["location_id", "latitude", "longitude"]
        ].isnull().sum().to_string()
    )

    print("\nUnique cities:", enriched["city"].nunique())

    print("\nSample enriched records:")
    print(
        enriched[
            [
                "city",
                "location_name",
                "sensor_id",
                "latitude",
                "longitude",
                "parameter",
                "value",
            ]
        ].head().to_string(index=False)
    )

    print("\nOutput file:", OUTPUT_FILE)
    print("\nMetadata enrichment completed.")


if __name__ == "__main__":
    main()
