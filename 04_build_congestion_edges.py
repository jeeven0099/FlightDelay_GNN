"""
STEP 4 — BUILD CONGESTION EDGES (DYNAMIC-READY)
=================================================
Builds the static congestion edge TOPOLOGY from 5-year BTS data.
Edge weights are computed dynamically per snapshot in step 5.

Three sub-signals:
  A. taxi_anomaly  — airports with historically correlated taxi delays
  B. volume        — high-traffic route pairs (structural flow capacity)
  C. hub_spoke     — hub→spoke and spoke→hub structural connections

The KEY difference from the old version:
  OLD: edge_weight = 5yr avg z-score (same weight every snapshot)
  NEW: edge topology saved here, weights computed live in step 5
       from actual taxi_out and dep_delay at each snapshot hour

Output: congestion_edges.parquet
  Columns: src_airport, dst_airport, edge_type, sub_type,
           base_weight, taxi_mean_origin, taxi_std_origin

Usage:
    python 04_build_congestion_edges.py
"""

import os
import glob
import numpy as np
import pandas as pd

# ── CONFIG ───────────────────────────────────────────────────────────────────
FLIGHTS_PATH        = r"C:\Users\user\Desktop\Airline_Graphs_Project"
OUTPUT_DIR          = r"C:\Users\user\Desktop\Airline_Graphs_Project\graph_data"
MIN_PAIR_FLIGHTS    = 50
HUB_TOP_N           = 30
TAXI_ZSCORE_THRESH  = 1.3   # lower = more edges fire
MIN_CONGESTED_HOURS = 10    # pair must co-congest this many hours to get an edge
# ─────────────────────────────────────────────────────────────────────────────


def find_flight_file(root):
    for ext in ("*.parquet", "*.csv", "*.feather"):
        m = glob.glob(os.path.join(root, ext))
        if m:
            return m[0]
    raise FileNotFoundError(f"No flight file in {root}")


def load_flights(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        df = pd.read_parquet(path)
    elif ext == ".csv":
        df = pd.read_csv(path, low_memory=False)
    else:
        df = pd.read_feather(path)
    for col in ("dep_datetime", "arr_datetime"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ("DepDelay","ArrDelay","TaxiOut","TaxiIn","Distance"):
        if col not in df.columns:
            df[col] = np.nan
    df = df.reset_index(drop=True)
    df["flight_id"] = df.index
    print(f"  Flights loaded: {len(df):,} rows")
    return df


# ── Per-airport taxi baseline (used in step 5 for live weight computation) ──

def build_taxi_baseline(df):
    """
    Compute per-airport taxi baseline (mean + std) for z-score normalisation.
    Step 5 uses this to compute live congestion weights per snapshot:
      z = (actual_taxi - mean) / std
      if z > threshold: congestion is active
    """
    print("\nBuilding taxi baseline per airport ...")
    baseline = (df.groupby("ORIGIN")["TaxiOut"]
                  .agg(taxi_mean="mean", taxi_std="std", taxi_n="count")
                  .reset_index()
                  .rename(columns={"ORIGIN":"airport"}))
    baseline["taxi_std"] = baseline["taxi_std"].fillna(5.0).clip(lower=1.0)
    print(f"  Baselines for {len(baseline)} airports")
    return baseline


# ── A. Taxi-anomaly edges (topology only) ────────────────────────────────────

def build_taxi_anomaly_edges(df, baseline):
    """
    For each (airport, hour) where taxi z-score > threshold, find all
    destination airports served. If two airports frequently co-congest
    (both show taxi anomalies in the same hours), add an edge.

    This gives the TOPOLOGY of which pairs co-congest.
    Step 5 computes actual weights per snapshot from live taxi data.
    """
    print("\n[A] Taxi-anomaly edges ...")
    if "TaxiOut" not in df.columns or "dep_datetime" not in df.columns:
        print("  Skipped — need TaxiOut and dep_datetime")
        return pd.DataFrame()

    sub = df[["ORIGIN","DEST","dep_datetime","TaxiOut"]].dropna()
    sub["dep_hour"] = sub["dep_datetime"].dt.floor("h")

    # Compute hourly taxi z-score per airport
    hourly = (sub.groupby(["ORIGIN","dep_hour"])
                 .agg(avg_taxi=("TaxiOut","mean"), n=("TaxiOut","count"))
                 .reset_index())
    hourly = hourly.merge(baseline.rename(columns={"airport":"ORIGIN"}),
                          on="ORIGIN", how="left")
    hourly["z"] = (hourly["avg_taxi"] - hourly["taxi_mean"]) / hourly["taxi_std"]
    congested = hourly[hourly["z"] > TAXI_ZSCORE_THRESH][["ORIGIN","dep_hour","z"]]

    print(f"  Congested airport-hours: {len(congested):,}")

    # For each congested hour, what destinations does that airport serve?
    sub2 = sub.merge(congested, on=["ORIGIN","dep_hour"], how="inner")

    # Edge: ORIGIN → DEST when ORIGIN is congested
    # Keep edges that co-occur at least MIN_CONGESTED_HOURS times
    edges = (sub2.groupby(["ORIGIN","DEST"])
                 .agg(co_congest_hours=("dep_hour","nunique"),
                      avg_zscore=("z","mean"))
                 .reset_index())
    edges = edges[edges["co_congest_hours"] >= MIN_CONGESTED_HOURS]

    # Also build DEST → ORIGIN reverse edges (congestion propagates both ways)
    rev = edges.rename(columns={"ORIGIN":"DEST","DEST":"ORIGIN"})
    rev = rev.rename(columns={"ORIGIN":"src_airport","DEST":"dst_airport"})
    fwd = edges.rename(columns={"ORIGIN":"src_airport","DEST":"dst_airport"})
    edges = pd.concat([fwd, rev], ignore_index=True)
    edges = edges[edges["src_airport"] != edges["dst_airport"]]

    # Merge in taxi baseline for src airport (needed by step 5)
    edges = edges.merge(
        baseline.rename(columns={"airport":"src_airport"}),
        on="src_airport", how="left")

    edges["edge_weight"] = (edges["avg_zscore"] / (TAXI_ZSCORE_THRESH * 3)).clip(0, 1)
    edges["edge_type"]   = "congestion"
    edges["sub_type"]    = "taxi_anomaly"

    print(f"  Taxi-anomaly edges: {len(edges):,} "
          f"(threshold: ≥{MIN_CONGESTED_HOURS} co-congested hours)")
    return edges


# ── B. Volume edges ──────────────────────────────────────────────────────────

def build_volume_edges(df):
    """
    High-traffic route pairs — these share gate/slot capacity and
    have structural flow dependency regardless of current congestion.
    """
    print("\n[B] Volume-based pair edges ...")
    pair_counts = (df.groupby(["ORIGIN","DEST"])
                     .size().reset_index(name="flight_count"))
    pair_counts = pair_counts[pair_counts["flight_count"] >= MIN_PAIR_FLIGHTS]
    max_count   = pair_counts["flight_count"].max()
    pair_counts["edge_weight"] = pair_counts["flight_count"] / max_count
    pair_counts["edge_type"]   = "congestion"
    pair_counts["sub_type"]    = "volume"
    pair_counts = pair_counts.rename(columns={"ORIGIN":"src_airport",
                                               "DEST":  "dst_airport"})
    print(f"  Volume edges (≥{MIN_PAIR_FLIGHTS} flights): {len(pair_counts):,}")
    return pair_counts


# ── C. Hub-spoke edges ────────────────────────────────────────────────────────

def build_hub_spoke_edges(df):
    """
    Hub airports transmit congestion to downstream spokes.
    A delayed hub blocks connecting passengers and resources.
    """
    print(f"\n[C] Hub-spoke edges (top {HUB_TOP_N} hubs) ...")
    dep_counts = df["ORIGIN"].value_counts()
    hubs       = set(dep_counts.head(HUB_TOP_N).index)

    spoke_pairs = df[df["ORIGIN"].isin(hubs) | df["DEST"].isin(hubs)].copy()
    hub_spoke   = []

    for hub in hubs:
        to_sp = (spoke_pairs[spoke_pairs["ORIGIN"]==hub]["DEST"]
                 .value_counts().reset_index())
        to_sp.columns = ["dst_airport","flight_count"]
        to_sp["src_airport"] = hub
        to_sp["direction"]   = "hub_to_spoke"
        hub_spoke.append(to_sp)

        fr_sp = (spoke_pairs[spoke_pairs["DEST"]==hub]["ORIGIN"]
                 .value_counts().reset_index())
        fr_sp.columns = ["src_airport","flight_count"]
        fr_sp["dst_airport"] = hub
        fr_sp["direction"]   = "spoke_to_hub"
        hub_spoke.append(fr_sp)

    hs_df = pd.concat(hub_spoke, ignore_index=True)
    hs_df = hs_df[hs_df["src_airport"] != hs_df["dst_airport"]]
    hs_df["edge_weight"] = (hs_df["flight_count"] /
                             hs_df["flight_count"].max()).clip(0, 1)
    hs_df["edge_type"]   = "congestion"
    hs_df["sub_type"]    = "hub_spoke"

    print(f"  Hub-spoke edges: {len(hs_df):,}")
    return hs_df


# ── Airport congestion stats (node features) ─────────────────────────────────

def compute_airport_stats(df, hubs):
    dep_counts  = df["ORIGIN"].value_counts()
    stats = (df.groupby("ORIGIN")
               .agg(total_departures=("flight_id","count"),
                    avg_dep_delay   =("DepDelay", "mean"),
                    p90_dep_delay   =("DepDelay", lambda x: x.quantile(0.9)),
                    avg_taxi_out    =("TaxiOut",  "mean"))
               .reset_index().rename(columns={"ORIGIN":"airport"}))
    arr_stats = (df.groupby("DEST")
                   .agg(avg_arr_delay=("ArrDelay","mean"))
                   .reset_index().rename(columns={"DEST":"airport"}))
    stats = stats.merge(arr_stats, on="airport", how="outer")
    stats["is_hub"] = stats["airport"].isin(hubs).astype(int)
    return stats


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("="*65)
    print("STEP 4 — BUILD CONGESTION EDGES (DYNAMIC-READY)")
    print("="*65)

    # Load the weather-enriched flights (same source as step 5)
    wx_path  = os.path.join(OUTPUT_DIR, "weather_node_features.parquet")
    raw_path = os.path.join(FLIGHTS_PATH, "flights_2018_2022.parquet")
    src = wx_path if os.path.exists(wx_path) else raw_path
    print(f"\nLoading flights from {os.path.basename(src)} ...")
    df  = load_flights(src)

    dep_counts = df.get("ORIGIN", df.get("Origin", pd.Series())).value_counts() \
                 if "ORIGIN" in df.columns else pd.Series()
    hubs = set(dep_counts.head(HUB_TOP_N).index)

    baseline   = build_taxi_baseline(df)

    # Build edge topologies
    taxi_edges = build_taxi_anomaly_edges(df, baseline)
    vol_edges  = build_volume_edges(df)
    hs_edges   = build_hub_spoke_edges(df)

    all_edges  = pd.concat([taxi_edges, vol_edges, hs_edges],
                            ignore_index=True).fillna(0)

    # Drop string columns that pyarrow can't serialize mixed with numerics
    drop_cols = ["direction", "co_congest_hours", "event_count", "avg_zscore",
                 "flight_count"]
    all_edges = all_edges.drop(
        columns=[c for c in drop_cols if c in all_edges.columns])

    # Ensure taxi baseline columns are present for step 5 live weighting
    for col in ["taxi_mean","taxi_std"]:
        if col not in all_edges.columns:
            all_edges[col] = 0.0

    print(f"\nTotal congestion edges: {len(all_edges):,}")
    for st in all_edges["sub_type"].unique():
        n = (all_edges["sub_type"]==st).sum()
        print(f"  {st:<15}: {n:,}")

    out = os.path.join(OUTPUT_DIR, "congestion_edges.parquet")
    all_edges.to_parquet(out, index=False)
    print(f"\n✅  Saved → {out}")

    # Save taxi baseline separately for step 5
    bl_out = os.path.join(OUTPUT_DIR, "taxi_baseline.parquet")
    baseline.to_parquet(bl_out, index=False)
    print(f"✅  Saved taxi baseline → {bl_out}")

    # Airport stats
    stats = compute_airport_stats(df, hubs)
    st_out = os.path.join(OUTPUT_DIR, "airport_congestion_stats.parquet")
    stats.to_parquet(st_out, index=False)
    print(f"✅  Saved airport stats → {st_out}")

    print("\nNext: run 05_build_graph_snapshots.py")
    print("      Congestion edge WEIGHTS are now computed dynamically")
    print("      in step 5 from actual per-date taxi data per snapshot.")


if __name__ == "__main__":
    main()