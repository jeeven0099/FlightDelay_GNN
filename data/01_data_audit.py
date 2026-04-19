"""
STEP 1 — DATA AUDIT
Run this first. It tells you:
  - Flight dataset shape, date range, airports, missing values
  - Which weather files exist and which airports/years they cover
  - Coverage overlap between flights and weather
  - Rotation chain statistics (how many tail-number chains exist)

Usage:
    python 01_data_audit.py

Edit the two path constants below to match your machine.
"""

import os
import glob
import re
from collections import defaultdict

import pandas as pd
import numpy as np

# ── EDIT THESE TWO PATHS ────────────────────────────────────────────────────
FLIGHTS_PATH   = r"C:\Users\user\Desktop\Airline_Graphs_Project"
WEATHER_DIR    = r"C:\Users\user\Desktop\Airline_Graphs_Project\data\datasets\noaa_weather_data\noaa_weather_parsed"
# ────────────────────────────────────────────────────────────────────────────

ROTATION_GAP_HOURS = 3          # max hours between legs to form a rotation edge
REPORT_TOP_N       = 20         # how many airports to show in summary tables


def find_flight_file(root: str) -> str:
    """Locate the flights CSV/parquet in the project root."""
    for ext in ("*.csv", "*.parquet", "*.feather"):
        matches = glob.glob(os.path.join(root, ext))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"No CSV/parquet/feather file found in {root}. "
        "Put your flights file there or update FLIGHTS_PATH."
    )


def load_flights(path: str) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print("LOADING FLIGHTS")
    print(f"{'='*60}")
    print(f"  File: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path, low_memory=False)
    elif ext == ".parquet":
        df = pd.read_parquet(path)
    elif ext in (".feather", ".ftr"):
        df = pd.read_feather(path)
    else:
        raise ValueError(f"Unsupported extension: {ext}")

    print(f"  Rows: {len(df):,}   Columns: {df.shape[1]}")

    # Parse datetime columns
    for col in ("dep_datetime", "arr_datetime", "weather_time", "FlightDate"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")

    return df


def audit_flights(df: pd.DataFrame):
    print(f"\n{'='*60}")
    print("FLIGHT DATASET SUMMARY")
    print(f"{'='*60}")

    # Date range
    if "dep_datetime" in df.columns:
        print(f"\n  Date range (dep_datetime): "
              f"{df['dep_datetime'].min()} → {df['dep_datetime'].max()}")
    if "FlightDate" in df.columns:
        print(f"  Date range (FlightDate):   "
              f"{df['FlightDate'].min()} → {df['FlightDate'].max()}")

    # Airport coverage
    origins = set(df["ORIGIN"].dropna().unique()) if "ORIGIN" in df.columns else set()
    dests   = set(df["DEST"].dropna().unique())   if "DEST"   in df.columns else set()
    airports = origins | dests
    print(f"\n  Unique ORIGIN airports : {len(origins)}")
    print(f"  Unique DEST   airports : {len(dests)}")
    print(f"  Total unique airports  : {len(airports)}")

    # Airlines
    if "Operating_Airline" in df.columns:
        print(f"  Unique airlines        : {df['Operating_Airline'].nunique()}")

    # Tail numbers
    if "Tail_Number" in df.columns:
        print(f"  Unique tail numbers    : {df['Tail_Number'].nunique()}")

    # Missing values
    print(f"\n  {'Column':<25} {'Missing':>8} {'%':>7}")
    print(f"  {'-'*45}")
    key_cols = ["dep_datetime", "arr_datetime", "DepDelay", "ArrDelay",
                "TaxiOut", "TaxiIn", "Tail_Number", "ORIGIN", "DEST"]
    for col in key_cols:
        if col in df.columns:
            n = df[col].isna().sum()
            pct = 100 * n / len(df)
            print(f"  {col:<25} {n:>8,} {pct:>6.1f}%")

    # Delay distribution
    if "ArrDelay" in df.columns:
        d = df["ArrDelay"].dropna()
        print(f"\n  ArrDelay stats (minutes):")
        print(f"    mean={d.mean():.1f}  median={d.median():.1f}  "
              f"std={d.std():.1f}  p95={d.quantile(0.95):.1f}  "
              f"p99={d.quantile(0.99):.1f}")
        print(f"    delayed (>15 min): {(d > 15).sum():,} "
              f"({100*(d>15).mean():.1f}%)")

    # Busiest airports
    if "ORIGIN" in df.columns:
        print(f"\n  Top {REPORT_TOP_N} busiest origin airports:")
        top = df["ORIGIN"].value_counts().head(REPORT_TOP_N)
        for ap, cnt in top.items():
            print(f"    {ap}: {cnt:,}")

    return airports


def audit_weather(weather_dir: str):
    print(f"\n{'='*60}")
    print("WEATHER FILES SUMMARY")
    print(f"{'='*60}")

    # Find all weather files
    all_files = []
    for root, _, files in os.walk(weather_dir):
        for f in files:
            if f.lower().endswith((".csv", ".csc")):   # handle typos like .csc
                all_files.append(os.path.join(root, f))

    print(f"\n  Total weather files found: {len(all_files)}")

    # Parse airport_year from filename
    pattern = re.compile(r"([A-Z]{3})_(\d{4})\.(csv|csc)", re.IGNORECASE)
    weather_index = {}   # (airport, year) → file path
    bad_names     = []

    for fp in all_files:
        fname = os.path.basename(fp)
        m = pattern.match(fname)
        if m:
            ap   = m.group(1).upper()
            year = int(m.group(2))
            weather_index[(ap, year)] = fp
        else:
            bad_names.append(fname)

    if bad_names:
        print(f"\n  Files with unexpected naming ({len(bad_names)}):")
        for b in bad_names[:10]:
            print(f"    {b}")

    # Summarise coverage
    weather_airports = sorted({ap for ap, _ in weather_index})
    weather_years    = sorted({yr for _, yr in weather_index})
    print(f"\n  Airports with weather data : {len(weather_airports)}")
    print(f"  Years covered              : {weather_years}")

    # Coverage matrix (rows=airports, cols=years)
    print(f"\n  Coverage matrix (✓ = file exists):")
    header = "  " + f"{'Airport':<8}" + "".join(f"{y:>6}" for y in weather_years)
    print(header)
    for ap in weather_airports[:REPORT_TOP_N]:
        row = f"  {ap:<8}" + "".join(
            f"{'  ✓':>6}" if (ap, y) in weather_index else f"{'  ✗':>6}"
            for y in weather_years
        )
        print(row)
    if len(weather_airports) > REPORT_TOP_N:
        print(f"  ... and {len(weather_airports) - REPORT_TOP_N} more airports")

    # Peek at one file to confirm schema
    if weather_index:
        sample_fp = next(iter(weather_index.values()))
        sample_df = pd.read_csv(sample_fp, nrows=3, low_memory=False)
        print(f"\n  Sample weather columns:")
        print(f"    {list(sample_df.columns)}")

    return set(weather_airports), weather_index


def audit_coverage(flight_airports: set, weather_airports: set):
    print(f"\n{'='*60}")
    print("COVERAGE OVERLAP — FLIGHTS vs WEATHER")
    print(f"{'='*60}")

    covered   = flight_airports & weather_airports
    no_weather = flight_airports - weather_airports
    weather_only = weather_airports - flight_airports

    print(f"\n  Flight airports WITH weather data    : {len(covered)}")
    print(f"  Flight airports WITHOUT weather data : {len(no_weather)}")
    print(f"  Weather-only airports (not in flights): {len(weather_only)}")

    if no_weather:
        print(f"\n  Flight airports missing weather (sample):")
        for ap in sorted(no_weather)[:30]:
            print(f"    {ap}", end="  ")
        print()


def audit_rotation_chains(df: pd.DataFrame):
    print(f"\n{'='*60}")
    print("AIRCRAFT ROTATION CHAIN ANALYSIS")
    print(f"{'='*60}")

    if "Tail_Number" not in df.columns or "dep_datetime" not in df.columns:
        print("  Skipped — need Tail_Number and dep_datetime columns.")
        return

    needed = ["Tail_Number", "ORIGIN", "DEST", "dep_datetime", "arr_datetime"]
    sub = df[needed].dropna(subset=["Tail_Number", "dep_datetime", "arr_datetime"])
    sub = sub.sort_values(["Tail_Number", "dep_datetime"])

    total_legs   = len(sub)
    chain_count  = 0
    gap_minutes  = []
    matching_airport = 0

    grouped = sub.groupby("Tail_Number")
    for tail, grp in grouped:
        rows = grp.reset_index(drop=True)
        for i in range(len(rows) - 1):
            curr = rows.iloc[i]
            nxt  = rows.iloc[i + 1]
            gap  = (nxt["dep_datetime"] - curr["arr_datetime"]).total_seconds() / 3600
            if 0 <= gap <= ROTATION_GAP_HOURS:
                chain_count += 1
                gap_minutes.append(gap * 60)
                if curr["DEST"] == nxt["ORIGIN"]:
                    matching_airport += 1

    print(f"\n  Total flight legs analysed      : {total_legs:,}")
    print(f"  Rotation edge candidates        : {chain_count:,}")
    print(f"    (gap ≤ {ROTATION_GAP_HOURS}h between arr and next dep)")
    print(f"  Matching airport (DEST=next ORIGIN): "
          f"{matching_airport:,} ({100*matching_airport/max(chain_count,1):.1f}%)")

    if gap_minutes:
        arr = np.array(gap_minutes)
        print(f"\n  Turnaround gap stats (minutes):")
        print(f"    mean={arr.mean():.1f}  median={np.median(arr):.1f}  "
              f"p25={np.percentile(arr,25):.1f}  p75={np.percentile(arr,75):.1f}")


def main():
    print("\n" + "★"*60)
    print("  FLIGHT DELAY GRAPH — DATA AUDIT")
    print("★"*60)

    # ── Flights ───────────────────────────────────────────────────
    flight_file = find_flight_file(FLIGHTS_PATH)
    df = load_flights(flight_file)
    flight_airports = audit_flights(df)
    audit_rotation_chains(df)

    # ── Weather ───────────────────────────────────────────────────
    weather_airports, weather_index = audit_weather(WEATHER_DIR)

    # ── Overlap ───────────────────────────────────────────────────
    audit_coverage(flight_airports, weather_airports)

    print(f"\n{'='*60}")
    print("AUDIT COMPLETE — review the output above before proceeding")
    print("Next: run  02_build_rotation_edges.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()