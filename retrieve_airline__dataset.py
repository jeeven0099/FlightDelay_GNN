import requests
import os
from tqdm import tqdm

# Years and months you want
years = [2022, 2023, 2024, 2025]
months = list(range(1,13))

output_dir = "bts_flight_data"
os.makedirs(output_dir, exist_ok=True)

url = "https://transtats.bts.gov/DownLoad_Table.asp"

for year in years:
    for month in months:
        print(f"Downloading {year}-{month:02d}")

        payload = {
            "Table_ID": "236",
            "Has_Group": "3",
            "Is_Zipped": "0",
            "DBShortName": "On_Time",
            "RawDataTable": "T_ONTIME",
            "Filter1": f"{year}",
            "Filter2": f"{month}"
        }

        response = requests.post(url, data=payload)

        filename = f"{output_dir}/flights_{year}_{month:02d}.csv"

        with open(filename, "wb") as f:
            f.write(response.content)

print("Download complete")