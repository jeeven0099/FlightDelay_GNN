"""
clean_and_merge_flights.py
Reads all 5 Kaggle BTS CSVs, keeps only the columns the GNN pipeline
needs, renames them to match the pipeline schema, builds dep_datetime
and arr_datetime, drops cancelled/diverted flights, and saves one
clean parquet file.

61 columns → 16 columns
~9 GB CSV → ~1.5 GB parquet

Usage:
    pip install pandas pyarrow tqdm
    python clean_and_merge_flights.py
"""

import os
import pandas as pd
from tqdm import tqdm

# ── CONFIG ──────────────────────────────────────────────────────────────────
DATA_DIR    = r"C:\Users\user\Downloads\archive (2)"
OUTPUT_FILE = r"C:\Users\user\Desktop\Airline_Graphs_Project\flights_2018_2022.parquet"

FILES = [
    "Combined_Flights_2018.csv",
    "Combined_Flights_2019.csv",
    "Combined_Flights_2020.csv",
    "Combined_Flights_2021.csv",
    "Combined_Flights_2022.csv",
]

# ── Columns to KEEP and how to rename them ───────────────────────────────────
#   Left  = name in the Kaggle CSV
#   Right = name the pipeline expects
KEEP_AND_RENAME = {
    "FlightDate"               : "FlightDate",
    "Operating_Airline"        : "Operating_Airline",   # IATA carrier code e.g. "C5"
    "Tail_Number"              : "Tail_Number",          # ← rotation edges
    "Origin"                   : "ORIGIN",
    "Dest"                     : "DEST",
    "DepDelay"                 : "DepDelay",
    "ArrDelay"                 : "ArrDelay",
    "TaxiOut"                  : "TaxiOut",
    "TaxiIn"                   : "TaxiIn",
    "AirTime"                  : "AirTime",
    "Distance"                 : "Distance",
    # Delay cause breakdown — useful as edge/node features later
    # (these don't exist in the Kaggle repack but we'll add NaN cols if missing)
    # Time fields to build dep_datetime / arr_datetime
    "CRSDepTime"               : "CRSDepTime",
    "DepTime"                  : "DepTime",
    "CRSArrTime"               : "CRSArrTime",
    "ArrTime"                  : "ArrTime",
    # Calendar fields — useful for time embeddings
    "Year"                     : "Year",
    "Month"                    : "Month",
    "DayofMonth"               : "DayofMonth",
    "DayOfWeek"                : "DayOfWeek",
}

# Columns to drop cancelled/diverted on
FILTER_COLS = ["Cancelled", "Diverted"]
# ────────────────────────────────────────────────────────────────────────────


def hhmm_to_timedelta(series: pd.Series) -> pd.Series:
    """
    Convert BTS integer time (e.g. 1435 → 14h 35min) to timedelta.
    Handles 2400 (midnight rollover) and NaN safely.
    """
    t = pd.to_numeric(series, errors="coerce").fillna(0).astype(int)
    # Clamp 2400 → 0 (some BTS rows use 2400 for midnight)
    t = t.where(t != 2400, 0)
    hours   = (t // 100).clip(0, 23)
    minutes = (t % 100).clip(0, 59)
    return pd.to_timedelta(hours, unit="h") + pd.to_timedelta(minutes, unit="m")


def process_file(filepath: str) -> pd.DataFrame:
    year = os.path.basename(filepath).split("_")[-1].replace(".csv", "")
    print(f"\n  [{year}] Loading {os.path.basename(filepath)} ...")

    # Read only the columns we want (faster + less RAM)
    usecols = list(KEEP_AND_RENAME.keys()) + FILTER_COLS
    # Some columns might be absent — only request what's actually there
    all_cols = pd.read_csv(filepath, nrows=0).columns.tolist()
    usecols  = [c for c in usecols if c in all_cols]

    df = pd.read_csv(filepath, usecols=usecols, low_memory=False)
    print(f"  [{year}] Loaded {len(df):,} rows")

    # ── 1. Drop cancelled and diverted ────────────────────────────────────
    before = len(df)
    if "Cancelled" in df.columns:
        df = df[df["Cancelled"] != True]
    if "Diverted" in df.columns:
        df = df[df["Diverted"] != True]
    df = df.drop(columns=[c for c in FILTER_COLS if c in df.columns])
    print(f"  [{year}] After drop cancelled/diverted: {len(df):,} "
          f"(removed {before - len(df):,})")

    # ── 2. Rename columns ─────────────────────────────────────────────────
    rename = {k: v for k, v in KEEP_AND_RENAME.items() if k in df.columns}
    df = df.rename(columns=rename)

    # ── 3. Build FlightDate as proper date ────────────────────────────────
    df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce")

    # ── 4. Build dep_datetime and arr_datetime ────────────────────────────
    dep_td = hhmm_to_timedelta(df["DepTime"])
    arr_td = hhmm_to_timedelta(df["ArrTime"])

    df["dep_datetime"] = df["FlightDate"] + dep_td
    df["arr_datetime"] = df["FlightDate"] + arr_td

    # Handle overnight flights: if arr < dep, arrival is next day
    overnight = df["arr_datetime"] < df["dep_datetime"]
    df.loc[overnight, "arr_datetime"] += pd.Timedelta(days=1)
    print(f"  [{year}] Overnight flights fixed: {overnight.sum():,}")

    # ── 5. Drop rows with no tail number, origin, dest, or datetime ───────
    before = len(df)
    df = df.dropna(subset=["Tail_Number", "ORIGIN", "DEST",
                            "dep_datetime", "arr_datetime"])
    # Drop rows where Tail_Number is "UNKNOW" or "00000" (BTS placeholders)
    df = df[~df["Tail_Number"].isin(["UNKNOW", "00000", "BLOCKED", ""])]
    print(f"  [{year}] After dropping invalid rows: {len(df):,} "
          f"(removed {before - len(df):,})")

    # ── 6. Drop the raw time integer columns (we have datetimes now) ───────
    df = df.drop(columns=["CRSDepTime", "DepTime", "CRSArrTime", "ArrTime"],
                 errors="ignore")

    # ── 7. Cast delay columns to float (some have NaN) ────────────────────
    for col in ["DepDelay", "ArrDelay", "TaxiOut", "TaxiIn", "AirTime", "Distance"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"  [{year}] Final columns: {list(df.columns)}")
    return df


def main():
    print("=" * 60)
    print("CLEAN AND MERGE FLIGHT DATA")
    print("=" * 60)
    print(f"Input : {DATA_DIR}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Files : {len(FILES)}")

    frames = []
    for fname in FILES:
        fp = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fp):
            print(f"\n  ⚠  Not found: {fp} — skipping")
            continue
        df = process_file(fp)
        frames.append(df)

    print(f"\n{'=' * 60}")
    print("CONCATENATING ALL YEARS ...")
    print("=" * 60)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("dep_datetime").reset_index(drop=True)

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n  Total rows      : {len(combined):,}")
    print(f"  Date range      : {combined['FlightDate'].min().date()} "
          f"→ {combined['FlightDate'].max().date()}")
    print(f"  Unique airports : {combined['ORIGIN'].nunique()}")
    print(f"  Unique tails    : {combined['Tail_Number'].nunique():,}")
    print(f"  Unique airlines : {combined['Operating_Airline'].nunique()}")
    print(f"\n  Columns kept ({len(combined.columns)}):")
    for col in combined.columns:
        null_pct = 100 * combined[col].isna().mean()
        print(f"    {col:<30} {null_pct:.1f}% null")

    # ── ArrDelay stats ──────────────────────────────────────────────────────
    d = combined["ArrDelay"].dropna()
    print(f"\n  ArrDelay stats (minutes):")
    print(f"    mean={d.mean():.1f}  median={d.median():.1f}  "
          f"std={d.std():.1f}  p95={d.quantile(0.95):.1f}")
    print(f"    delayed (>15 min): {(d > 15).sum():,} ({100*(d>15).mean():.1f}%)")

    # ── Save ───────────────────────────────────────────────────────────────
    print(f"\nSaving to parquet ...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    combined.to_parquet(OUTPUT_FILE, index=False)

    size_gb = os.path.getsize(OUTPUT_FILE) / 1e9
    print(f"\n✅  Saved → {OUTPUT_FILE}  ({size_gb:.2f} GB)")
    print("\nNext: run  01_data_audit.py")
    print("      (make sure FLIGHTS_PATH points to the project root)")


if __name__ == "__main__":
    main()