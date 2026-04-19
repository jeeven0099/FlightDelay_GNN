import requests
import os
import gzip
import csv
import time
from io import StringIO

# Airport USAF station IDs
stations = {
    "ATL": "722190",
    "DFW": "722590",
    "DEN": "725650",
    "ORD": "725300",
    "LAX": "722950",
    "JFK": "744860",
    "LAS": "723860",
    "MCO": "722050",
    "SEA": "727930",
    "CLT": "723140",
    "MIA": "722020",
    "PHX": "722780",
    "IAH": "722430",
    "EWR": "725020",
    "SFO": "724940",
    "MSP": "726580",
    "DTW": "725370",
    "BOS": "725090",
    "PHL": "724080",
    "LGA": "725030",
    "BWI": "724060",
    "SLC": "724720",
    "TPA": "722220",
    "SAN": "722900",
    "FLL": "722070",
    "BNA": "723270",
    "PDX": "727950",
    "AUS": "722460",
    "IAD": "724030",
    "MCI": "724330",
    "IND": "724380",
    "PIT": "725205",
    "CLE": "725130",
    "RDU": "723130",
    "SJC": "724945",
    "OAK": "724980",
    "MKE": "726400",
    "HOU": "722440",
    "ANC": "702610",
    "CMH": "724220",
}

years = [2018, 2019, 2020, 2021, 2022]
base_url = "https://www.ncei.noaa.gov/pub/data/noaa/"
ISD_HISTORY_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"

output_dir = "noaa_weather_data"
parsed_dir = "noaa_weather_parsed"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(parsed_dir, exist_ok=True)

MAX_RETRIES = 3
RETRY_DELAY = 5

# ── Step 1: Download ISD history and build USAF → WBAN map ──────────────────
print("Fetching ISD station history to resolve WBAN IDs...")
r = requests.get(ISD_HISTORY_URL, timeout=30)
r.raise_for_status()

# isd-history.csv columns:
# USAF, WBAN, STATION NAME, CTRY, ST, ICAO, LAT, LON, ELEV(M), BEGIN, END
usaf_to_wbans = {}  # USAF -> list of WBAN candidates
reader = csv.DictReader(StringIO(r.text))
for row in reader:
    usaf = row["USAF"].strip()
    wban = row["WBAN"].strip()
    if usaf not in usaf_to_wbans:
        usaf_to_wbans[usaf] = []
    usaf_to_wbans[usaf].append(wban)

print(f"Loaded {len(usaf_to_wbans)} stations from ISD history.\n")


def resolve_wban(usaf, year):
    """
    Try each known WBAN for a USAF ID and return the first one that
    produces a 200 response for the given year. Falls back to '99999'.
    """
    candidates = usaf_to_wbans.get(usaf, [])
    # Always try 99999 first (common default), then real WBANs
    ordered = ["99999"] + [w for w in candidates if w != "99999"]
    for wban in ordered:
        filename = f"{usaf}-{wban}-{year}.gz"
        url = base_url + f"{year}/{filename}"
        try:
            head = requests.head(url, timeout=10)
            if head.status_code == 200:
                return wban, url
        except requests.RequestException:
            continue
    return None, None


# ── Step 2: Resolve all USAF→WBAN pairs up front ────────────────────────────
print("Resolving WBAN IDs (HEAD requests)...")
resolved = {}  # airport -> {year -> (wban, url)}
for airport, usaf in stations.items():
    resolved[airport] = {}
    for year in years:
        wban, url = resolve_wban(usaf, year)
        if wban:
            resolved[airport][year] = (wban, url)
            print(f"  {airport} {year}: USAF={usaf} WBAN={wban}  ✓")
        else:
            print(f"  {airport} {year}: USAF={usaf} — no valid file found  ✗")
print()


# ── Step 3: ISD record parser ────────────────────────────────────────────────
def parse_isd_record(line):
    if len(line) < 60:
        return None

    def scale(val, factor, missing):
        return None if val == missing else val / factor

    try:
        rec = {}
        rec["station_id"]   = line[4:10].strip()
        rec["wban"]         = line[10:15].strip()
        rec["year"]         = int(line[15:19])
        rec["month"]        = int(line[19:21])
        rec["day"]          = int(line[21:23])
        rec["hour"]         = int(line[23:25])
        rec["minute"]       = int(line[25:27])
        rec["datetime"]     = (f"{rec['year']}-{rec['month']:02d}-{rec['day']:02d} "
                               f"{rec['hour']:02d}:{rec['minute']:02d}")

        lat_raw = int(line[28:34])
        lon_raw = int(line[34:41])
        rec["latitude"]     = scale(lat_raw, 1000, 99999)
        rec["longitude"]    = scale(lon_raw, 1000, 999999)

        elev_raw = int(line[46:51])
        rec["elevation_m"]  = elev_raw if elev_raw != 9999 else None

        wind_dir = int(line[60:63])
        rec["wind_dir_deg"] = wind_dir if wind_dir != 999 else None

        wind_spd = int(line[65:69])
        rec["wind_speed_ms"] = scale(wind_spd, 10, 9999)

        ceil_raw = int(line[70:75])
        rec["ceiling_m"]    = ceil_raw if ceil_raw != 99999 else None

        vis_raw = int(line[78:84])
        rec["visibility_m"] = vis_raw if vis_raw != 999999 else None

        temp_raw = int(line[87:92])
        rec["temp_c"]       = scale(temp_raw, 10, 9999)

        dew_raw = int(line[93:98])
        rec["dewpoint_c"]   = scale(dew_raw, 10, 9999)

        slp_raw = int(line[99:104])
        rec["sea_level_pressure_hpa"] = scale(slp_raw, 10, 99999)

        # Additional data sections
        add = line[105:].strip() if len(line) > 105 else ""

        def get_add(tag):
            idx = add.find(tag)
            return add[idx:] if idx != -1 else None

        # Precipitation (AA1)
        aa = get_add("AA1")
        if aa and len(aa) >= 8:
            try:
                rec["precip_depth_mm"] = scale(int(aa[3:7]), 10, 9999)
                rec["precip_period_h"] = int(aa[1:3]) if aa[1:3].strip().isdigit() else None
            except ValueError:
                rec["precip_depth_mm"] = rec["precip_period_h"] = None
        else:
            rec["precip_depth_mm"] = rec["precip_period_h"] = None

        # Snow depth (AJ1)
        aj = get_add("AJ1")
        if aj and len(aj) >= 7:
            try:
                v = int(aj[3:7])
                rec["snow_depth_mm"] = v if v != 9999 else None
            except ValueError:
                rec["snow_depth_mm"] = None
        else:
            rec["snow_depth_mm"] = None

        # Sky cover (GF1)
        gf = get_add("GF1")
        if gf and len(gf) >= 5:
            try:
                v = int(gf[3:5])
                rec["sky_cover_oktas"] = v if v != 99 else None
            except ValueError:
                rec["sky_cover_oktas"] = None
        else:
            rec["sky_cover_oktas"] = None

        # Pressure: altimeter + station (MA1)
        ma = get_add("MA1")
        if ma and len(ma) >= 13:
            try:
                rec["altimeter_hpa"]       = scale(int(ma[1:6]),  10, 99999)
                rec["station_pressure_hpa"]= scale(int(ma[7:12]), 10, 99999)
            except ValueError:
                rec["altimeter_hpa"] = rec["station_pressure_hpa"] = None
        else:
            rec["altimeter_hpa"] = rec["station_pressure_hpa"] = None

        # Pressure tendency (MD1)
        md = get_add("MD1")
        if md and len(md) >= 8:
            try:
                rec["pressure_tendency_hpa"] = scale(int(md[2:7]), 10, 99999)
            except ValueError:
                rec["pressure_tendency_hpa"] = None
        else:
            rec["pressure_tendency_hpa"] = None

        # Wind gust (OC1)
        oc = get_add("OC1")
        if oc and len(oc) >= 7:
            try:
                rec["wind_gust_ms"] = scale(int(oc[3:7]), 10, 9999)
            except ValueError:
                rec["wind_gust_ms"] = None
        else:
            rec["wind_gust_ms"] = None

        # Extreme temp (KA1)
        ka = get_add("KA1")
        if ka and len(ka) >= 9:
            try:
                rec["extreme_temp_c"] = scale(int(ka[4:9]), 10, 9999)
            except ValueError:
                rec["extreme_temp_c"] = None
        else:
            rec["extreme_temp_c"] = None

        # Derived relative humidity (Magnus formula)
        T, Td = rec.get("temp_c"), rec.get("dewpoint_c")
        if T is not None and Td is not None:
            import math
            gamma = (17.625 * Td) / (243.04 + Td) - (17.625 * T) / (243.04 + T)
            rec["relative_humidity_pct"] = round(100 * math.exp(gamma), 1)
        else:
            rec["relative_humidity_pct"] = None

        return rec
    except (ValueError, IndexError):
        return None


FIELDS = [
    "datetime", "year", "month", "day", "hour", "minute",
    "latitude", "longitude", "elevation_m",
    "temp_c", "dewpoint_c", "relative_humidity_pct",
    "wind_dir_deg", "wind_speed_ms", "wind_gust_ms",
    "sea_level_pressure_hpa", "altimeter_hpa", "station_pressure_hpa",
    "pressure_tendency_hpa",
    "visibility_m", "ceiling_m", "sky_cover_oktas",
    "precip_depth_mm", "precip_period_h", "snow_depth_mm", "extreme_temp_c",
    "station_id", "wban",
]

# ── Step 4: Download & parse ─────────────────────────────────────────────────
failed = []

for airport, usaf in stations.items():
    for year in years:
        if year not in resolved[airport]:
            failed.append((airport, year))
            continue

        wban, url = resolved[airport][year]
        gz_path  = f"{output_dir}/{airport}_{year}.gz"
        csv_path = f"{parsed_dir}/{airport}_{year}.csv"

        success = False
        for attempt in range(1, MAX_RETRIES + 1):
            print(f"Downloading {airport} ({usaf}-{wban}) {year}  [attempt {attempt}]")
            try:
                r = requests.get(url, timeout=60)
                if r.status_code == 200:
                    with open(gz_path, "wb") as f:
                        f.write(r.content)
                    success = True
                    break
                else:
                    print(f"  HTTP {r.status_code}, retrying...")
                    time.sleep(RETRY_DELAY)
            except requests.RequestException as e:
                print(f"  Request error: {e}, retrying...")
                time.sleep(RETRY_DELAY)

        if not success:
            failed.append((airport, year))
            continue

        try:
            with gzip.open(gz_path, "rt", encoding="latin-1", errors="replace") as f:
                records = [r for line in f if (r := parse_isd_record(line))]

            with open(csv_path, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=FIELDS, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(records)

            print(f"  ✓ {len(records)} records → {csv_path}")
        except Exception as e:
            print(f"  ✗ Parse error: {e}")
            failed.append((airport, year))

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
if failed:
    print(f"Failed ({len(failed)}):")
    for airport, year in failed:
        print(f"  {airport} {year}")
else:
    print("All airports downloaded and parsed successfully.")