"""
download_bts_flights.py
=======================
Downloads the BTS Airline On-Time Performance dataset directly from
https://www.transtats.bts.gov for every month in 2018–2024.

Each downloaded zip contains one CSV with ALL the columns you need:
  FL_DATE, TAIL_NUM (= Tail_Number), UNIQUE_CARRIER (= airline),
  ORIGIN, DEST, DEP_DELAY, ARR_DELAY, TAXI_OUT, TAXI_IN,
  CRS_DEP_TIME, DEP_TIME, ARR_TIME, AIR_TIME, DISTANCE,
  CARRIER_DELAY, WEATHER_DELAY, NAS_DELAY, SECURITY_DELAY,
  LATE_AIRCRAFT_DELAY

After downloading, this script also unzips and concatenates everything
into a single  flights_2018_2024.parquet  file ready for the pipeline.

Usage:
    pip install requests tqdm pandas pyarrow
    python download_bts_flights.py

The script resumes safely — already-downloaded zip files are skipped.
Expected total size: ~8 GB compressed, ~30 GB uncompressed.
Final parquet: ~4–6 GB (efficient columnar compression).
"""

import os
import time
import zipfile
import glob
import requests
import pandas as pd
from tqdm import tqdm

# ── CONFIG ──────────────────────────────────────────────────────────────────
DOWNLOAD_DIR = r"C:\Users\user\Desktop\Airline_Graphs_Project\bts_raw"
OUTPUT_FILE  = r"C:\Users\user\Desktop\Airline_Graphs_Project\flights_2018_2024.parquet"

START_YEAR   = 2018
END_YEAR     = 2024   # inclusive
SLEEP_SEC    = 2      # polite delay between requests
# ────────────────────────────────────────────────────────────────────────────

# BTS POST endpoint (this is the actual form submission target)
BTS_URL = "https://www.transtats.bts.gov/DownLoad_Table.asp"

# Exact field list to request — these are the BTS internal column names
# Every field maps directly to what the GNN pipeline expects
BTS_FIELDS = [
    "YEAR", "QUARTER", "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK",
    "FL_DATE",          # → FlightDate
    "UNIQUE_CARRIER",   # → Operating_Airline
    "TAIL_NUM",         # → Tail_Number  ← critical for rotation edges
    "FL_NUM",
    "ORIGIN",
    "DEST",
    "CRS_DEP_TIME",
    "DEP_TIME",
    "DEP_DELAY",        # → DepDelay
    "TAXI_OUT",         # → TaxiOut
    "WHEELS_OFF",
    "WHEELS_ON",
    "TAXI_IN",          # → TaxiIn
    "CRS_ARR_TIME",
    "ARR_TIME",
    "ARR_DELAY",        # → ArrDelay
    "CANCELLED",
    "CANCELLATION_CODE",
    "DIVERTED",
    "AIR_TIME",         # → AirTime
    "DISTANCE",         # → Distance
    "CARRIER_DELAY",    # delay cause breakdown
    "WEATHER_DELAY",
    "NAS_DELAY",
    "SECURITY_DELAY",
    "LATE_AIRCRAFT_DELAY",
]

# Build the SELECT clause
SQL_FIELDS = ",".join(BTS_FIELDS)


def build_post_payload(year: int, month: int) -> dict:
    """
    Constructs the POST payload that mimics the BTS web form submission.
    The sqlstr format was reverse-engineered from the BTS Transtats form.
    """
    sqlstr = (
        f"+SELECT+{SQL_FIELDS}"
        f"+FROM++T_ONTIME_REPORTING"
        f"+WHERE+Month+%3D{month}+AND+YEAR%3D{year}"
    )
    return {
        "UserTableName"  : "On_Time_Reporting",
        "DBShortName"    : "On_Time",
        "RawDataTable"   : "T_ONTIME_REPORTING",
        "sqlstr"         : sqlstr,
        "varlist"        : SQL_FIELDS,
        "grouplist"      : "",
        "suml"           : "",
        "sumRegion"      : "",
        "filter1"        : "title",
        "filter2"        : "title",
        "geo"            : "All",
        "time"           : str(month),
        "timename"       : "Month",
        "GEOGRAPHY"      : "All",
        "XYEAR"          : str(year),
        "FREQUENCY"      : str(month),
        "VarDesc"        : "",
        "TABLE"          : "",
        "Out_OPT"        : "D",      # D = download
        "ButtonOpt"      : "Download",
    }


def download_month(year: int, month: int, download_dir: str, session: requests.Session) -> str | None:
    """
    Download one month's data. Returns the zip filepath on success, None if skipped.
    """
    fname    = f"bts_{year}_{month:02d}.zip"
    out_path = os.path.join(download_dir, fname)

    if os.path.exists(out_path):
        print(f"  ↩  {fname} already exists — skipping")
        return out_path

    payload = build_post_payload(year, month)

    try:
        resp = session.post(
            BTS_URL,
            data=payload,
            timeout=120,
            headers={
                "User-Agent"  : "Mozilla/5.0 (research data download)",
                "Referer"     : "https://www.transtats.bts.gov/DL_SelectFields.aspx",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            stream=True,
        )
        resp.raise_for_status()

        # BTS returns zip if successful, HTML error page if failed
        content_type = resp.headers.get("Content-Type", "")
        if "zip" not in content_type and "octet" not in content_type:
            print(f"  ❌  {fname} — unexpected Content-Type: {content_type}")
            print(f"      Response preview: {resp.text[:200]}")
            return None

        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)

        size_mb = os.path.getsize(out_path) / 1e6
        print(f"  ✅  {fname} downloaded ({size_mb:.1f} MB)")
        return out_path

    except requests.RequestException as e:
        print(f"  ❌  {fname} — request failed: {e}")
        return None


def extract_zip(zip_path: str, extract_dir: str) -> list[str]:
    """Unzip and return list of extracted CSV paths."""
    csv_paths = []
    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            if name.endswith(".csv"):
                z.extract(name, extract_dir)
                csv_paths.append(os.path.join(extract_dir, name))
    return csv_paths


RENAME_MAP = {
    "FL_DATE"          : "FlightDate",
    "UNIQUE_CARRIER"   : "Operating_Airline",
    "TAIL_NUM"         : "Tail_Number",
    "FL_NUM"           : "FlightNum",
    "DEP_DELAY"        : "DepDelay",
    "ARR_DELAY"        : "ArrDelay",
    "TAXI_OUT"         : "TaxiOut",
    "TAXI_IN"          : "TaxiIn",
    "AIR_TIME"         : "AirTime",
    "DISTANCE"         : "Distance",
    "CANCELLED"        : "Cancelled",
    "DIVERTED"         : "Diverted",
    "CARRIER_DELAY"    : "CarrierDelay",
    "WEATHER_DELAY"    : "WeatherDelay",
    "NAS_DELAY"        : "NASDelay",
    "SECURITY_DELAY"   : "SecurityDelay",
    "LATE_AIRCRAFT_DELAY": "LateAircraftDelay",
}

KEEP_COLS = [
    "FlightDate", "Operating_Airline", "Tail_Number", "FlightNum",
    "ORIGIN", "DEST",
    "CRS_DEP_TIME", "DEP_TIME", "DepDelay",
    "TaxiOut", "WHEELS_OFF", "WHEELS_ON", "TaxiIn",
    "CRS_ARR_TIME", "ARR_TIME", "ArrDelay",
    "Cancelled", "Diverted",
    "AirTime", "Distance",
    "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay",
    "YEAR", "MONTH", "DAY_OF_MONTH",
]


def clean_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)

    # Drop the trailing empty column BTS sometimes adds
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    df = df.rename(columns=RENAME_MAP)

    # Build dep_datetime and arr_datetime from FL_DATE + DEP_TIME / ARR_TIME
    df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce")

    def time_to_hhmm(t):
        """Convert BTS integer time (e.g. 1435 → '14:35') to timedelta."""
        t = pd.to_numeric(t, errors="coerce").fillna(0).astype(int)
        hours   = (t // 100).clip(0, 23)
        minutes = (t % 100).clip(0, 59)
        return pd.to_timedelta(hours, unit="h") + pd.to_timedelta(minutes, unit="m")

    if "DEP_TIME" in df.columns:
        df["dep_datetime"] = df["FlightDate"] + time_to_hhmm(df["DEP_TIME"])
    if "ARR_TIME" in df.columns:
        df["arr_datetime"] = df["FlightDate"] + time_to_hhmm(df["ARR_TIME"])
        # Handle overnight flights: if arr < dep, add 1 day
        if "dep_datetime" in df.columns:
            overnight = df["arr_datetime"] < df["dep_datetime"]
            df.loc[overnight, "arr_datetime"] += pd.Timedelta(days=1)

    # Keep only relevant columns
    actual_keep = [c for c in KEEP_COLS + ["dep_datetime", "arr_datetime"] if c in df.columns]
    df = df[actual_keep]

    # Drop cancelled flights (no delay info useful for graph)
    if "Cancelled" in df.columns:
        df = df[df["Cancelled"] != 1.0]

    # Drop rows with no ORIGIN or DEST or Tail_Number
    df = df.dropna(subset=["ORIGIN", "DEST", "Tail_Number"])

    return df


def main():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    session = requests.Session()

    # ── Phase 1: Download all zips ─────────────────────────────────────────
    print("=" * 60)
    print("PHASE 1 — DOWNLOADING FROM BTS TRANSTATS")
    print("=" * 60)
    print(f"Years: {START_YEAR}–{END_YEAR}  ({(END_YEAR - START_YEAR + 1) * 12} monthly files)")
    print(f"Output dir: {DOWNLOAD_DIR}\n")

    zip_paths = []
    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            print(f"  {year}-{month:02d} ...", end=" ")
            zp = download_month(year, month, DOWNLOAD_DIR, session)
            if zp:
                zip_paths.append(zp)
            time.sleep(SLEEP_SEC)

    # ── Phase 2: Extract and clean ─────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("PHASE 2 — EXTRACTING AND CLEANING")
    print("=" * 60)

    csv_dir = os.path.join(DOWNLOAD_DIR, "csv")
    os.makedirs(csv_dir, exist_ok=True)

    frames = []
    for zp in tqdm(zip_paths, desc="Processing zips"):
        csv_paths = extract_zip(zp, csv_dir)
        for cp in csv_paths:
            try:
                df = clean_csv(cp)
                frames.append(df)
            except Exception as e:
                print(f"\n  ⚠  Failed to process {cp}: {e}")

    if not frames:
        print("❌  No data frames loaded. Check download step.")
        return

    # ── Phase 3: Concatenate and save ──────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("PHASE 3 — CONCATENATING AND SAVING")
    print("=" * 60)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("dep_datetime").reset_index(drop=True)

    print(f"\n  Total rows    : {len(combined):,}")
    print(f"  Date range    : {combined['FlightDate'].min()} → {combined['FlightDate'].max()}")
    print(f"  Unique airports: {combined['ORIGIN'].nunique()}")
    print(f"  Tail numbers  : {combined['Tail_Number'].nunique():,}")

    combined.to_parquet(OUTPUT_FILE, index=False)
    size_gb = os.path.getsize(OUTPUT_FILE) / 1e9
    print(f"\n✅  Saved → {OUTPUT_FILE}  ({size_gb:.2f} GB)")
    print("\nNow update FLIGHTS_PATH in the pipeline scripts to point to this file's directory.")
    print("The audit script will auto-detect flights_2018_2024.parquet.")


if __name__ == "__main__":
    main()