
from src.data_collection.openaq_collector import (
    collect_all_city_locations
)

def main():
    print("Starting AIRSIGHT AI data collection...\n")

    locations = collect_all_city_locations()

    print("\nTotal locations collected:", len(locations))

    for city in [
        "Delhi", "Mumbai", "Ahmedabad", "Bengaluru",
        "Chennai", "Kolkata", "Hyderabad", "Pune"
    ]:
        count = sum(
            1 for location in locations
            if location.get("target_city") == city
        )
        print(f"{city}: {count} locations")


if __name__ == "__main__":
    main()
