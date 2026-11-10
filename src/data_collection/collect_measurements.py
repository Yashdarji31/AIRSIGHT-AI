import csv
import time
import requests

from datetime import datetime, timedelta, timezone

from src.config import (
    OPENAQ_API_KEY,
    OPENAQ_BASE_URL,
    RAW_DATA_DIR,
)

SENSOR_FILE = RAW_DATA_DIR / "openaq_sensors.csv"
OUTPUT_FILE = RAW_DATA_DIR / "openaq_hourly_measurements.csv"

HEADERS = {"X-API-Key": OPENAQ_API_KEY}

# Six main pollutants for the first pilot
POLLUTANTS = ["pm25", "pm10", "no2", "so2", "co", "o3"]

# Pilot configuration
DAYS_TO_COLLECT = 7
MAX_SENSORS_PER_CITY_PARAMETER = 1
PAGE_LIMIT = 1000


def load_sensors():
    """Load sensor metadata from the discovery CSV."""
    with open(SENSOR_FILE, "r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def select_pilot_sensors(sensors):
    """Select the most recently active sensor per city and pollutant."""
    selected = []

    cities = sorted({
        sensor["target_city"]
        for sensor in sensors
        if sensor.get("target_city")
    })

    for city in cities:
        for pollutant in POLLUTANTS:
            candidates = [
                sensor for sensor in sensors
                if sensor["target_city"] == city
                and sensor["parameter_name"] == pollutant
                and sensor.get("sensor_id")
                and sensor.get("last_measurement_utc")
            ]

            candidates.sort(
                key=lambda sensor: sensor["last_measurement_utc"],
                reverse=True
            )

            selected.extend(
                candidates[:MAX_SENSORS_PER_CITY_PARAMETER]
            )

    return selected


def collect_sensor_hours(sensor):
    """Retrieve hourly measurements for one sensor, including pagination."""
    sensor_id = sensor["sensor_id"]

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=DAYS_TO_COLLECT)

    url = f"{OPENAQ_BASE_URL}/sensors/{sensor_id}/hours"

    measurements = []
    page = 1

    while True:
        params = {
            "datetime_from": start_time.isoformat(),
            "datetime_to": end_time.isoformat(),
            "limit": PAGE_LIMIT,
            "page": page,
        }

        response = requests.get(
            url,
            headers=HEADERS,
            params=params,
            timeout=45
        )

        response.raise_for_status()

        result = response.json()
        page_results = result.get("results", [])

        if not page_results:
            break

        measurements.extend(page_results)

        meta = result.get("meta", {})
        found = meta.get("found")

        if found is not None:
            if page * PAGE_LIMIT >= int(found):
                break
        elif len(page_results) < PAGE_LIMIT:
            break

        page += 1

    return measurements


def main():
    print("Starting AIRSIGHT AI hourly measurement collection...\n")

    sensors = load_sensors()
    selected_sensors = select_pilot_sensors(sensors)

    print("Total sensors selected:", len(selected_sensors))
    print("Collection period:", DAYS_TO_COLLECT, "days\n")

    all_records = []
    successful_sensors = 0
    failed_sensors = 0

    for index, sensor in enumerate(selected_sensors, start=1):
        city = sensor["target_city"]
        pollutant = sensor["parameter_name"]
        sensor_id = sensor["sensor_id"]

        print(
            f"[{index}/{len(selected_sensors)}] "
            f"{city} | {pollutant} | Sensor {sensor_id}"
        )

        try:
            measurements = collect_sensor_hours(sensor)

            print("    Records retrieved:", len(measurements))

            for measurement in measurements:
                period = measurement.get("period") or {}
                datetime_from = period.get("datetimeFrom") or {}
                datetime_to = period.get("datetimeTo") or {}

                coordinates = measurement.get("coordinates") or {}
                coverage = measurement.get("coverage") or {}

                all_records.append({
                    "sensor_id": sensor_id,
                    "location_name": sensor.get("location_name"),
                    "city": city,
                    "parameter": pollutant,
                    "unit": (
                        measurement.get("parameter", {}).get("units")
                        or sensor.get("parameter_units")
                    ),
                    "value": measurement.get("value"),
                    "datetime_from_utc": datetime_from.get("utc"),
                    "datetime_to_utc": datetime_to.get("utc"),
                    "latitude": coordinates.get("latitude"),
                    "longitude": coordinates.get("longitude"),
                    "coverage_percent": coverage.get("percentCoverage"),
                })

            successful_sensors += 1

        except requests.RequestException as error:
            failed_sensors += 1
            print("    Request failed:", error)

        except (ValueError, KeyError, TypeError) as error:
            failed_sensors += 1
            print("    Data processing failed:", error)

        time.sleep(0.2)

    if all_records:
        fieldnames = list(all_records[0].keys())

        with open(
            OUTPUT_FILE,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_records)

    print("\n========== COLLECTION SUMMARY ==========")
    print("Sensors selected:", len(selected_sensors))
    print("Successful sensors:", successful_sensors)
    print("Failed sensors:", failed_sensors)
    print("Total hourly records:", len(all_records))
    print("Output file:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
