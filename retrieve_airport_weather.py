import requests
import os

# Airport weather station IDs
stations = {
    # Top passenger traffic & highly connected
    "ATL": "722190",  # Atlanta Hartsfield-Jackson
    "DFW": "722590",  # Dallas/Fort Worth
    "DEN": "725650",  # Denver
    "ORD": "725300",  # Chicago O'Hare
    "LAX": "722950",  # Los Angeles
    "JFK": "744860",  # New York JFK
    "LAS": "723860",  # Las Vegas
    "MCO": "722050",  # Orlando
    "SEA": "727930",  # Seattle-Tacoma
    "CLT": "723140",  # Charlotte Douglas
    "MIA": "722020",  # Miami
    "PHX": "722780",  # Phoenix Sky Harbor
    "IAH": "722430",  # Houston Bush Intercontinental
    "EWR": "725020",  # Newark
    "SFO": "724940",  # San Francisco
    "MSP": "726580",  # Minneapolis-St Paul
    "DTW": "725370",  # Detroit
    "BOS": "725090",  # Boston Logan
    "PHL": "724080",  # Philadelphia
    "LGA": "725030",  # New York LaGuardia

    # Additional highly connected hubs / top connections
    "BWI": "724060",  # Baltimore/Washington
    "SLC": "724720",  # Salt Lake City
    "TPA": "722220",  # Tampa Bay
    "SAN": "722900",  # San Diego
    "FLL": "722070",  # Fort Lauderdale
    "BNA": "724110",  # Nashville
    "PDX": "727950",  # Portland
    "AUS": "722460",  # Austin
    "IAD": "724030",  # Washington Dulles
    "MCI": "724330",  # Kansas City
    "IND": "724110",  # Indianapolis
    "PIT": "724040",  # Pittsburgh
    "CLE": "725130",  # Cleveland
    "RDU": "723130",  # Raleigh-Durham
    "SJC": "724940",  # San Jose
    "OAK": "724980",  # Oakland
    "MKE": "726580",  # Milwaukee
    "HOU": "722440",  # Houston Hobby
    "ANC": "702610",  # Anchorage
    "CMH": "724220",  # Columbus
}

years = [2022, 2023, 2024, 2025]
base_url = "https://www.ncei.noaa.gov/pub/data/noaa/"

output_dir = "noaa_weather_data"
os.makedirs(output_dir, exist_ok=True)

for airport, station in stations.items():
    for year in years:
        filename = f"{station}-99999-{year}.gz"
        url = base_url + f"{year}/{filename}"

        print(f"Downloading {airport} {year}")

        r = requests.get(url)

        if r.status_code == 200:
            with open(f"{output_dir}/{airport}_{year}.gz", "wb") as f:
                f.write(r.content)

print("Weather data download complete")