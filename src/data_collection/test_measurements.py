
import csv
from datetime import datetime, timedelta, timezone

import requests

from src.config import OPENAQ_API_KEY, OPENAQ_BASE_URL, RAW_DATA_DIR

SENSOR_FILE = RAW_DATA_DIR / "openaq_sensors.csv"
HEADERS = {"X-API-Key": OPENAQ_API_KEY}


def main():
    # Load sensor metadata
    with open(SENSOR_FILE, "r", encoding="utf-8") as file:
        sensors = list(csv.DictReader(file))

    # Find PM2.5 sensors with recorded measurements
    candidates = [
        sensor for sensor in sensors
        if sensor["parameter_name"] == "pm25"
        and sensor["sensor_id"]
        and sensor["last_measurement_utc"]
    ]

    if not candidates:
        print("No suitable PM2.5 sensor found.")
        return

    # Sort by most recent recorded measurement
    candidates.sort(
        key=lambda sensor: sensor["last_measurement_utc"],
        reverse=True
    )

    sensor = candidates[0]

    sensor_id = sensor["sensor_id"]

    print("Testing one PM2.5 sensor")
    print("City:", sensor["target_city"])
    print("Location:", sensor["location_name"])
    print("Sensor ID:", sensor_id)
    print("Unit:", sensor["parameter_units"])
    print("Last recorded measurement:", sensor["last_measurement_utc"])

    # Request the last 7 days of hourly measurements
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=7)

    params = {
        "datetime_from": start_time.isoformat(),
        "datetime_to": end_time.isoformat(),
        "limit": 1000,
        "page": 1,
    }

    url = f"{OPENAQ_BASE_URL}/sensors/{sensor_id}/hours"

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            params=params,
            timeout=30
        )

        print("\nHTTP status:", response.status_code)

        response.raise_for_status()

        result = response.json()
        measurements = result.get("results", [])

        print("Measurements retrieved:", len(measurements))

        for measurement in measurements[:5]:
            period = measurement.get("period") or {}
            datetime_from = period.get("datetimeFrom") or {}

            print({
                "value": measurement.get("value"),
                "datetime": datetime_from.get("utc"),
                "parameter": (measurement.get("parameter") or {}).get("name"),
                "unit": (measurement.get("parameter") or {}).get("units"),
            })

    except requests.RequestException as error:
        print("API request failed:", error)


if __name__ == "__main__":
    main()
