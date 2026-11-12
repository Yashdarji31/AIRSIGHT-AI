import csv
import time
import requests

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import (
    OPENAQ_API_KEY,
    OPENAQ_BASE_URL,
    RAW_DATA_DIR,
)

from src.data_collection.openaq_collector import (
    collect_all_city_locations,
)


# API authentication
HEADERS = {
    "X-API-Key": OPENAQ_API_KEY
}

# Save sensor metadata here
OUTPUT_FILE = RAW_DATA_DIR / "openaq_sensors.csv"


def create_session():
    """
    Create an HTTP session with retry handling.
    Retries temporary API errors and rate limits.
    """
    retry_strategy = Retry(
        total=5,
        connect=3,
        read=3,
        status=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
        respect_retry_after_header=True,
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)

    session = requests.Session()
    session.headers.update(HEADERS)
    session.mount("https://", adapter)

    return session


def collect_location_sensors(session, location):
    """
    Retrieve all available sensors for one OpenAQ location.
    """
    location_id = location.get("id")

    if location_id is None:
        print("Skipping location without an ID.")
        return []

    url = f"{OPENAQ_BASE_URL}/locations/{location_id}/sensors"

    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()

        data = response.json()
        sensors = data.get("results", [])

        sensor_records = []

        for sensor in sensors:
            parameter = sensor.get("parameter") or {}
            coverage = sensor.get("coverage") or {}
            latest = sensor.get("latest") or {}
            latest_datetime = latest.get("datetime") or {}
            latest_coordinates = latest.get("coordinates") or {}
            coordinates = sensor.get("coordinates") or {}

            record = {
                "target_city": location.get("target_city"),
                "location_id": location_id,
                "location_name": location.get("name"),

                "sensor_id": sensor.get("id"),
                "sensor_name": sensor.get("name"),

                "parameter_id": parameter.get("id"),
                "parameter_name": parameter.get("name"),
                "parameter_units": parameter.get("units"),
                "parameter_display_name": parameter.get("displayName"),

                "first_measurement_utc": (
                    (sensor.get("datetimeFirst") or {}).get("utc")
                ),
                "last_measurement_utc": (
                    (sensor.get("datetimeLast") or {}).get("utc")
                ),

                "percent_coverage": coverage.get("percentCoverage"),
                "percent_complete": coverage.get("percentComplete"),
                "observed_count": coverage.get("observedCount"),

                "latest_value": latest.get("value"),
                "latest_datetime_utc": latest_datetime.get("utc"),

                "latitude": (
                    coordinates.get("latitude")
                    or latest_coordinates.get("latitude")
                ),
                "longitude": (
                    coordinates.get("longitude")
                    or latest_coordinates.get("longitude")
                ),
            }

            sensor_records.append(record)

        return sensor_records

    except requests.RequestException as error:
        print(
            f"Error retrieving sensors for location "
            f"{location_id}: {error}"
        )
        return []


def save_sensor_data(sensor_records):
    """
    Save discovered sensor metadata to CSV.
    """
    if not sensor_records:
        print("No sensor records available to save.")
        return

    fieldnames = list(sensor_records[0].keys())

    with open(
        OUTPUT_FILE,
        mode="w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(sensor_records)

    print(f"\nSensor data saved to: {OUTPUT_FILE}")


def discover_all_sensors():
    """
    Discover sensors across all target cities.
    """
    print("Starting AIRSIGHT AI sensor discovery...\n")

    # Reuse the existing location collector
    locations = collect_all_city_locations()

    # Avoid requesting the same location more than once
    unique_locations = {}
    for location in locations:
        location_id = location.get("id")

        if location_id is not None:
            unique_locations.setdefault(location_id, location)

    locations = list(unique_locations.values())

    print(f"\nUnique locations to process: {len(locations)}")

    session = create_session()
    all_sensor_records = []

    try:
        for index, location in enumerate(locations, start=1):

            city = location.get("target_city", "Unknown")
            location_name = location.get("name", "Unknown")

            print(
                f"[{index}/{len(locations)}] "
                f"{city} - {location_name}"
            )

            sensors = collect_location_sensors(
                session,
                location,
            )

            for sensor in sensors:
                all_sensor_records.append(sensor)

            print(f"    Sensors discovered: {len(sensors)}")

            # Small delay between location requests
            time.sleep(0.2)

    finally:
        session.close()

    save_sensor_data(all_sensor_records)

    print("\n========== SENSOR DISCOVERY SUMMARY ==========")
    print(f"Locations processed: {len(locations)}")
    print(f"Total sensors discovered: {len(all_sensor_records)}")

    print("\nSensors by city:")

    for city in sorted({
        location.get("target_city", "Unknown")
        for location in locations
    }):
        city_count = sum(
            1 for sensor in all_sensor_records
            if sensor.get("target_city") == city
        )

        print(f"{city}: {city_count} sensors")

    print("\nSensors by pollutant:")

    pollutant_counts = {}

    for sensor in all_sensor_records:
        pollutant = sensor.get("parameter_name") or "Unknown"

        pollutant_counts[pollutant] = (
            pollutant_counts.get(pollutant, 0) + 1
        )

    for pollutant, count in sorted(pollutant_counts.items()):
        print(f"{pollutant}: {count}")

    print("\nSensor discovery completed!")


if __name__ == "__main__":
    discover_all_sensors()
