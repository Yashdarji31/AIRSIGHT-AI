
import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENAQ_API_KEY")

if not API_KEY:
    raise ValueError("OpenAQ API key not found!")

url = "https://api.openaq.org/v3/locations"

headers = {
    "X-API-Key": API_KEY
}

params = {
    "iso": "IN",
    "limit": 5,
    "page": 1
}

response = requests.get(
    url,
    headers=headers,
    params=params,
    timeout=30
)

print("Status Code:", response.status_code)

response.raise_for_status()

data = response.json()

print("\nOpenAQ API connected successfully!")

for location in data.get("results", []):
    print(location.get("name"))
