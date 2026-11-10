import requests
from src.config import (
    OPENAQ_API_KEY,
    OPENAQ_BASE_URL,
    CITIES,
)

HEADERS = {
    "X-API-Key": OPENAQ_API_KEY
}


def collect_city_locations(city_name, coordinates):
    """
    Retrieve OpenAQ monitoring locations near a city.
    """

    url = f"{OPENAQ_BASE_URL}/locations"

    params = {
        "iso": "IN",
        "coordinates": f"{coordinates['latitude']},{coordinates['longitude']}",
        "radius": 25000,
        "limit": 100,
        "page": 1,
    }

    response = requests.get(
        url,
        headers=HEADERS,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    results = response.json().get("results", [])

    print(f"{city_name}: {len(results)} locations retrieved")

    return results


def collect_all_city_locations():
    """
    Retrieve monitoring locations for all configured cities.
    """

    all_locations = []

    for city_name, coordinates in CITIES.items():
        try:
            locations = collect_city_locations(
                city_name,
                coordinates,
            )

            for location in locations:
                location["target_city"] = city_name

            all_locations.extend(locations)

        except requests.RequestException as error:
            print(f"Error collecting {city_name}: {error}")

    return all_locations
