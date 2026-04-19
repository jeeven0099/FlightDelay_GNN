"""
DATASET VALIDATION — Flight Delay GNN
======================================
Runs sanity checks on the raw flight data and graph data files.
Catches common issues: wrong delay distributions, missing tail numbers,
impossible values, date range problems, airport coverage gaps.

Run locally on your machine (no GPU needed):
    python validate_dataset.py

All paths point to local Airline_Graphs_Project folder.
"""

import os
import numpy as np
import pandas as pd

# ── CONFIG ──────────────────────────────────────────────────────────────────
BASE_DIR       = r"C:\Users\user\Desktop\Airline_Graphs_Project"
GRAPH_DATA_DIR = os.path.join(BASE_DIR, "graph_data")
FLIGHTS_PATH   = os.path.join(GRAPH_DATA_DIR, "weather_node_features.parquet")

PASS = "  ✅"
FAIL = "  ❌"
WARN = "  ⚠️"

passed = failed = warned = 0

def check(name, condition, detail="", warn_only=False):
    global passed, failed, warned
    if condition:
        print(f"{PASS} {name}")
        if detail:
            print(f"      {detail}")
        passed += 1
    elif warn_only:
        print(f"{WARN} {name}")
        if detail:
            print(f"      {detail}")
        warned += 1
    else:
        print(f"{FAIL} {name}")
        if detail:
            print(f"      {detail}")
        failed += 1
# ────────────────────────────────────────────────────────────────────────────


def validate_flights(df):
    print("\n" + "="*60)
    print("1. FLIGHT DATA BASICS")
    print("="*60)

    # Row count
    check("Row count reasonable",
          5_000_000 < len(df) < 30_000_000,
          f"{len(df):,} rows")

    # Required columns
    required = ["ORIGIN","DEST","DepDelay","ArrDelay",
                "dep_datetime","arr_datetime","Tail_Number"]
    missing = [c for c in required if c not in df.columns]
    check("Required columns present",
          len(missing) == 0,
          f"Missing: {missing}" if missing else
          f"All {len(required)} required columns present")

    # Date range
    df["dep_datetime"] = pd.to_datetime(df["dep_datetime"], errors="coerce")
    min_yr = df["dep_datetime"].dt.year.min()
    max_yr = df["dep_datetime"].dt.year.max()
    check("Date range 2018-2022",
          min_yr == 2018 and max_yr == 2022,
          f"Actual range: {min_yr} → {max_yr}")

    # Year distribution
    yr_counts = df["dep_datetime"].dt.year.value_counts().sort_index()
    print(f"\n  Flights per year:")
    for yr, cnt in yr_counts.items():
        bar = "█" * int(cnt / 200_000)
        flag = "" if 1_000_000 < cnt < 3_000_000 else " ← unusual"
        print(f"    {yr}: {cnt:>9,} {bar}{flag}")

    # Null rates
    print(f"\n  Null rates on key columns:")
    for col in ["DepDelay","ArrDelay","Tail_Number","dep_datetime","arr_datetime"]:
        if col in df.columns:
            null_pct = df[col].isna().mean() * 100
            flag = PASS if null_pct < 5 else (WARN if null_pct < 15 else FAIL)
            print(f"    {col:<25} {null_pct:.1f}% null  {flag}")


def validate_delay_distributions(df):
    print("\n" + "="*60)
    print("2. DELAY DISTRIBUTIONS")
    print("="*60)

    arr = df["ArrDelay"].dropna()
    dep = df["DepDelay"].dropna()

    # ArrDelay stats
    print(f"\n  ArrDelay statistics:")
    print(f"    Mean   : {arr.mean():.1f} min")
    print(f"    Median : {arr.median():.1f} min")
    print(f"    Std    : {arr.std():.1f} min")
    print(f"    Min    : {arr.min():.0f} min")
    print(f"    Max    : {arr.max():.0f} min")
    print(f"    p25    : {arr.quantile(0.25):.1f} min")
    print(f"    p75    : {arr.quantile(0.75):.1f} min")
    print(f"    p95    : {arr.quantile(0.95):.1f} min")
    print(f"    p99    : {arr.quantile(0.99):.1f} min")

    # Sanity checks on distributions
    check("ArrDelay mean between 0 and 20 min",
          0 <= arr.mean() <= 20,
          f"Mean = {arr.mean():.1f} min (expected 4-10 for US domestic)")

    check("ArrDelay has early arrivals (negative values)",
          (arr < 0).mean() > 0.3,
          f"{(arr<0).mean()*100:.1f}% flights arrived early (expect 30-45%)")

    check("ArrDelay max not absurdly large",
          arr.max() < 2000,
          f"Max = {arr.max():.0f} min",
          warn_only=True)

    check("No impossible negative delays > 200 min",
          (arr < -200).mean() < 0.001,
          f"{(arr<-200).sum():,} flights with ArrDelay < -200 min",
          warn_only=True)

    # Delay rate
    pct_delayed_15 = (arr >= 15).mean() * 100
    pct_delayed_60 = (arr >= 60).mean() * 100
    print(f"\n  Delay rates:")
    print(f"    Delayed >15 min : {pct_delayed_15:.1f}%  (expect 15-25%)")
    print(f"    Delayed >60 min : {pct_delayed_60:.1f}%  (expect 4-8%)")
    print(f"    On time (<15min): {100-pct_delayed_15:.1f}%")

    check("Delay rate >15min between 10% and 35%",
          10 <= pct_delayed_15 <= 35,
          f"Actual: {pct_delayed_15:.1f}%")

    # DepDelay vs ArrDelay correlation
    both_valid = df[["DepDelay","ArrDelay"]].dropna()
    corr = both_valid["DepDelay"].corr(both_valid["ArrDelay"])
    check("DepDelay and ArrDelay are correlated",
          corr > 0.7,
          f"Correlation = {corr:.3f} (expect > 0.7)")


def validate_airports(df):
    print("\n" + "="*60)
    print("3. AIRPORT COVERAGE")
    print("="*60)

    expected_weather_airports = {
        'ANC','ATL','BNA','BOS','BWI','CLE','CLT','CMH','DEN','DFW',
        'DTW','EWR','FLL','HOU','IAD','IAH','IND','JFK','LAS','LAX',
        'LGA','MCI','MCO','MIA','MKE','MSP','ORD','PHL','PHX','PIT',
        'SAN','SEA','SFO','SJC','SLC','TPA'
    }

    # Filter to weather airports
    filtered = df[df["ORIGIN"].isin(expected_weather_airports) &
                  df["DEST"].isin(expected_weather_airports)]

    check("Filtered dataset has 9M+ rows",
          len(filtered) > 8_000_000,
          f"{len(filtered):,} rows after 36-airport filter")

    # Top 10 routes
    route_counts = (filtered.groupby(["ORIGIN","DEST"])
                             .size().nlargest(10))
    print(f"\n  Top 10 busiest routes:")
    for (o,d), cnt in route_counts.items():
        print(f"    {o}→{d}: {cnt:,}")

    # Airport coverage
    origins_covered = set(filtered["ORIGIN"].unique())
    missing_origins = expected_weather_airports - origins_covered
    check("All 36 weather airports appear as origins",
          len(missing_origins) == 0,
          f"Missing: {missing_origins}" if missing_origins else
          "All 36 airports have departures")

    # Flights per airport
    ap_counts = filtered.groupby("ORIGIN").size().sort_values(ascending=False)
    print(f"\n  Top 5 airports by departure count:")
    for ap, cnt in ap_counts.head(5).items():
        print(f"    {ap}: {cnt:,}")
    print(f"  Bottom 5:")
    for ap, cnt in ap_counts.tail(5).items():
        print(f"    {ap}: {cnt:,}")

    check("No airport has suspiciously few flights",
          ap_counts.min() > 10_000,
          f"Min flights at any airport: {ap_counts.min():,} ({ap_counts.idxmin()})")


def validate_tail_numbers(df):
    print("\n" + "="*60)
    print("4. TAIL NUMBER VALIDATION")
    print("="*60)

    tails = df["Tail_Number"].dropna()

    # Basic stats
    n_unique = tails.nunique()
    n_null   = df["Tail_Number"].isna().sum()
    null_pct = n_null / len(df) * 100

    print(f"  Unique tail numbers : {n_unique:,}")
    print(f"  Missing tail numbers: {n_null:,} ({null_pct:.1f}%)")

    check("Tail number null rate < 5%",
          null_pct < 5,
          f"Null rate: {null_pct:.1f}%")

    check("Reasonable number of unique tails",
          3000 < n_unique < 20000,
          f"Unique tails: {n_unique:,} (expect 4,000-8,000 for US domestic)")

    # Tail number format check (US N-numbers)
    valid_format = tails.str.match(r'^N\d').mean()
    check("Tail numbers look like US N-numbers",
          valid_format > 0.8,
          f"{valid_format*100:.1f}% start with 'N' (US registration format)")

    # Flights per tail per day
    df_copy = df[df["Tail_Number"].notna()].copy()
    df_copy["dep_date"] = pd.to_datetime(
        df_copy["dep_datetime"], errors="coerce").dt.date
    legs_per_tail_day = (df_copy.groupby(["Tail_Number","dep_date"])
                                .size())
    print(f"\n  Legs per tail per day:")
    print(f"    Mean  : {legs_per_tail_day.mean():.2f}")
    print(f"    Median: {legs_per_tail_day.median():.1f}")
    print(f"    Max   : {legs_per_tail_day.max()}")
    print(f"    >6 legs/day: {(legs_per_tail_day>6).sum():,} tail-days")

    check("Median legs per tail per day between 2 and 5",
          2 <= legs_per_tail_day.median() <= 5,
          f"Median: {legs_per_tail_day.median():.1f}")


def validate_temporal(df):
    print("\n" + "="*60)
    print("5. TEMPORAL CONSISTENCY")
    print("="*60)

    df["dep_dt"] = pd.to_datetime(df["dep_datetime"], errors="coerce")
    df["arr_dt"] = pd.to_datetime(df["arr_datetime"], errors="coerce")

    both_valid = df[df["dep_dt"].notna() & df["arr_dt"].notna()].copy()
    flight_time_min = ((both_valid["arr_dt"] - both_valid["dep_dt"])
                       .dt.total_seconds() / 60)

    print(f"  Flight duration statistics:")
    print(f"    Mean   : {flight_time_min.mean():.0f} min")
    print(f"    Min    : {flight_time_min.min():.0f} min")
    print(f"    Max    : {flight_time_min.max():.0f} min")
    print(f"    <0 min : {(flight_time_min<0).sum():,} (impossible)")
    print(f"    <30 min: {(flight_time_min<30).sum():,} (suspiciously short)")
    print(f"    >600min: {(flight_time_min>600).sum():,} (very long)")

    check("No negative flight durations",
          (flight_time_min < 0).mean() < 0.001,
          f"{(flight_time_min<0).sum():,} flights with negative duration",
          warn_only=True)

    check("Mean flight duration between 90 and 200 min",
          90 <= flight_time_min.mean() <= 200,
          f"Mean: {flight_time_min.mean():.0f} min")

    # Monthly seasonality — delays should be higher in summer
    df["month"] = df["dep_dt"].dt.month
    monthly_delay = df.groupby("month")["ArrDelay"].mean()
    print(f"\n  Average ArrDelay by month:")
    for mo, val in monthly_delay.items():
        bar = "█" * max(0, int((val + 5) / 2))
        print(f"    Month {mo:>2}: {val:>6.1f} min  {bar}")

    summer_avg = monthly_delay[[6,7,8]].mean()
    winter_avg = monthly_delay[[12,1,2]].mean()
    check("Summer delays higher than winter (seasonal pattern)",
          summer_avg > winter_avg,
          f"Summer avg: {summer_avg:.1f} min | Winter avg: {winter_avg:.1f} min",
          warn_only=True)


def validate_graph_files():
    print("\n" + "="*60)
    print("6. GRAPH DATA FILES")
    print("="*60)

    files = {
        "rotation_edges.parquet"  : (10_000_000, "rotation edges"),
        "congestion_edges.parquet": (5_000,       "congestion edges"),
        "airport_locations.parquet":(30,          "airport locations"),
        "network_edges.parquet"   : (1_000,       "network edges"),
        "flight_lookup.parquet"   : (5_000_000,   "flight lookup"),
    }

    for fname, (min_rows, desc) in files.items():
        fpath = os.path.join(GRAPH_DATA_DIR, fname)
        if os.path.exists(fpath):
            df_f = pd.read_parquet(fpath)
            check(f"{fname} has sufficient rows",
                  len(df_f) >= min_rows,
                  f"{len(df_f):,} rows ({desc})")
        else:
            check(f"{fname} exists", False,
                  f"File not found: {fpath}")

    # Check snapshot files exist
    for name in ["snapshots_train.pt","snapshots_val.pt","snapshots_test.pt",
                 "static_edges.pt"]:
        fpath = os.path.join(GRAPH_DATA_DIR, name)
        exists = os.path.exists(fpath)
        size_gb = os.path.getsize(fpath)/1e9 if exists else 0
        check(f"{name} exists",
              exists,
              f"{size_gb:.2f} GB" if exists else "File not found")


def validate_rotation_edges():
    print("\n" + "="*60)
    print("7. ROTATION EDGE VALIDATION")
    print("="*60)

    fpath = os.path.join(GRAPH_DATA_DIR, "rotation_edges.parquet")
    if not os.path.exists(fpath):
        print(f"{FAIL} rotation_edges.parquet not found")
        return

    re = pd.read_parquet(fpath)
    print(f"  Total rotation edges: {len(re):,}")

    # Turnaround time distribution
    if "turnaround_min" in re.columns:
        ta = re["turnaround_min"].dropna()
        print(f"\n  Turnaround time:")
        print(f"    Mean  : {ta.mean():.1f} min")
        print(f"    Median: {ta.median():.1f} min")
        print(f"    <20min: {(ta<20).mean()*100:.1f}% (very tight)")
        print(f"    <45min: {(ta<45).mean()*100:.1f}% (tight)")

        check("Mean turnaround > 30 min",
              ta.mean() > 30,
              f"Mean turnaround: {ta.mean():.1f} min")

    # Delay propagation signal
    if "leg1_arr_delay" in re.columns:
        delayed = (re["leg1_arr_delay"] > 15).mean() * 100
        print(f"\n  Rotation edges with inbound delay >15min: {delayed:.1f}%")
        check("Some rotation edges carry delay signal",
              delayed > 5,
              f"{delayed:.1f}% of rotations have inbound delay >15min")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("DATASET VALIDATION — Flight Delay GNN")
    print("=" * 60)

    # Load flights
    print(f"\nLoading flights from {FLIGHTS_PATH} ...")
    if not os.path.exists(FLIGHTS_PATH):
        alt = os.path.join(BASE_DIR, "flights_2018_2022.parquet")
        if os.path.exists(alt):
            print(f"  Using raw flights: {alt}")
            df = pd.read_parquet(alt)
        else:
            print(f"  ❌ Could not find flights file")
            return
    else:
        df = pd.read_parquet(FLIGHTS_PATH)

    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Run all checks
    validate_flights(df)
    validate_delay_distributions(df)
    validate_airports(df)
    validate_tail_numbers(df)
    validate_temporal(df)
    validate_graph_files()
    validate_rotation_edges()

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    total = passed + failed + warned
    print(f"  ✅ Passed  : {passed}")
    print(f"  ❌ Failed  : {failed}")
    print(f"  ⚠️  Warnings: {warned}")
    print(f"  Total     : {total}")

    if failed == 0:
        print(f"\n  Dataset looks clean — ready for modeling.")
    elif failed <= 2:
        print(f"\n  Minor issues found — review failures above.")
    else:
        print(f"\n  Multiple failures — investigate before trusting results.")


if __name__ == "__main__":
    main()