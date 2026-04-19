"""
STEP 2 — BUILD AIRCRAFT ROTATION EDGES
For every tail number, find consecutive flight pairs where:
  - arrival airport of leg N == departure airport of leg N+1
  - turnaround gap is between MIN_GAP_MIN and MAX_GAP_MIN minutes

Output: rotation_edges.parquet
  Columns:
    tail_number, leg1_flight_id, leg2_flight_id,
    origin, midpoint, dest,
    leg1_dep, leg1_arr, leg2_dep, leg2_arr,
    turnaround_min,
    leg1_arr_delay, leg1_dep_delay,   ← source delay signal
    edge_weight                         ← how "risky" this rotation is

Usage:
    python 02_build_rotation_edges.py
"""

import os, glob
import pandas as pd
import numpy as np

# ── CONFIG ──────────────────────────────────────────────────────────────────
FLIGHTS_PATH      = r"C:\Users\user\Desktop\Airline_Graphs_Project"
OUTPUT_DIR        = r"C:\Users\user\Desktop\Airline_Graphs_Project\graph_data"

MIN_GAP_MIN       = 0      # ignore legs where next dep is before arr (data error)
MAX_GAP_MIN       = 720    # 12-hour turnaround window (change 1: extended from 3h)
                             # captures same-day tail propagation across longer ground stops
                             # signal weakens with distance but remains meaningful to ~6h
# ────────────────────────────────────────────────────────────────────────────


def find_flight_file(root):
    for ext in ("*.csv", "*.parquet", "*.feather"):
        m = glob.glob(os.path.join(root, ext))
        if m:
            return m[0]
    raise FileNotFoundError(f"No flight file in {root}")


def load_flights(path):
    print(f"Loading flights from {path} ...")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path, low_memory=False)
    elif ext == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_feather(path)

    for col in ("dep_datetime", "arr_datetime", "FlightDate"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")

    # Create a stable flight index
    df = df.reset_index(drop=True)
    df["flight_id"] = df.index
    print(f"  {len(df):,} rows loaded.")
    return df


def build_rotation_edges(df):
    print("\nBuilding rotation edges ...")

    needed = ["flight_id", "Tail_Number", "ORIGIN", "DEST",
              "dep_datetime", "arr_datetime"]
    optional = ["DepDelay", "ArrDelay", "TaxiOut", "TaxiIn",
                "Operating_Airline", "Distance"]

    cols = needed + [c for c in optional if c in df.columns]
    sub = df[cols].dropna(subset=["Tail_Number", "dep_datetime", "arr_datetime"])
    sub = sub.sort_values(["Tail_Number", "dep_datetime"]).reset_index(drop=True)

    # Build in chunks to avoid OOM with 12h cap (~22M+ edges)
    CHUNK_SIZE = 500_000
    chunks     = []
    records    = []
    total      = 0
    tails      = sub.groupby("Tail_Number", sort=False)

    for tail, grp in tails:
        grp = grp.reset_index(drop=True)
        for i in range(len(grp) - 1):
            curr = grp.iloc[i]
            nxt  = grp.iloc[i + 1]

            gap_min = (nxt["dep_datetime"] - curr["arr_datetime"]).total_seconds() / 60
            if gap_min < MIN_GAP_MIN or gap_min > MAX_GAP_MIN:
                continue

            airport_match = (curr["DEST"] == nxt["ORIGIN"])

            # Fix 1: enforce airport match — skip teleportation edges
            # (same tail but plane appears at different airport = data error)
            if not airport_match:
                continue

            records.append({
                "edge_type"       : "rotation",
                "tail_number"     : tail,
                "leg1_flight_id"  : curr["flight_id"],
                "leg2_flight_id"  : nxt["flight_id"],
                "src_airport"     : curr["ORIGIN"],
                "mid_airport"     : curr["DEST"],
                "dst_airport"     : nxt["DEST"],
                "airport_match"   : True,   # always True (enforced above)
                "leg1_dep"        : curr["dep_datetime"],
                "leg1_arr"        : curr["arr_datetime"],
                "leg2_dep"        : nxt["dep_datetime"],
                "leg2_arr"        : nxt["arr_datetime"],
                "turnaround_min"  : round(gap_min, 2),
                "leg1_arr_delay"  : curr.get("ArrDelay", np.nan),
                "leg1_dep_delay"  : curr.get("DepDelay", np.nan),
                "leg2_dep_delay"  : nxt.get("DepDelay",  np.nan),
                "leg2_arr_delay"  : nxt.get("ArrDelay",  np.nan),
            })

            # Flush to chunk every CHUNK_SIZE records to keep memory low
            if len(records) >= CHUNK_SIZE:
                chunks.append(pd.DataFrame(records))
                total += len(records)
                print(f"    ... {total:,} edges so far")
                records = []

    # Flush remaining
    if records:
        chunks.append(pd.DataFrame(records))
        total += len(records)

    edges = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    print(f"  Total rotation edges: {len(edges):,}")

    if len(edges) == 0:
        print("  ⚠️  No edges found — check date parsing and column names.")
        return edges

    # ── Edge weight ────────────────────────────────────────────────────────
    # Higher weight = higher risk of delay propagation.
    # Formula: sigmoid( leg1_arr_delay / (turnaround_min + 1) )
    # Meaning: if a delayed plane has very little turnaround time, weight → 1
    arr_delay     = edges["leg1_arr_delay"].fillna(0).clip(lower=0)
    turnaround    = edges["turnaround_min"].clip(lower=1)
    risk_ratio    = arr_delay / turnaround
    edges["edge_weight"] = 1 / (1 + np.exp(-risk_ratio + 1))   # shifted sigmoid

    # ── Stats ──────────────────────────────────────────────────────────────
    print(f"\n  Airport match (DEST == next ORIGIN): "
          f"{edges['airport_match'].sum():,} / {len(edges):,} "
          f"({100*edges['airport_match'].mean():.1f}%)")
    print(f"\n  Turnaround gap (minutes):")
    print(edges["turnaround_min"].describe().to_string())
    print(f"\n  Edge weight distribution:")
    print(edges["edge_weight"].describe().to_string())

    delayed_src = (edges["leg1_arr_delay"] > 15).sum()
    print(f"\n  Rotation edges where leg1 was delayed (>15 min): "
          f"{delayed_src:,} ({100*delayed_src/len(edges):.1f}%)")

    return edges


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # CRITICAL: must load the same file as step 5 with the same row ordering
    # so that flight_id (df.index) matches between rotation_edges and snapshots.
    # Step 5 loads weather_node_features.parquet first, falls back to flights_2018_2022.parquet.
    # We do the same here — same priority order, same file = same row indices.
    wx_path  = os.path.join(OUTPUT_DIR, "weather_node_features.parquet")
    raw_path = os.path.join(FLIGHTS_PATH, "flights_2018_2022.parquet")

    if os.path.exists(wx_path):
        flight_file = wx_path
        print(f"Loading from weather_node_features.parquet (matches step 5)")
    elif os.path.exists(raw_path):
        flight_file = raw_path
        print(f"Loading from flights_2018_2022.parquet")
    else:
        flight_file = find_flight_file(FLIGHTS_PATH)
        print(f"Loading from {flight_file}")

    df    = load_flights(flight_file)
    edges = build_rotation_edges(df)

    if len(edges) > 0:
        out_path = os.path.join(OUTPUT_DIR, "rotation_edges.parquet")
        edges.to_parquet(out_path, index=False)
        print(f"\n✅  Saved → {out_path}")
    else:
        print("\n❌  No edges to save.")

    print("\nNext: run  03_build_weather_edges.py")


if __name__ == "__main__":
    main()