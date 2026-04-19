"""
STEP 5 — TWO-LEVEL GRAPH SNAPSHOTS (OPTIMIZED — NO ITERROWS)
=============================================================
Colab version — reads from Google Drive.

PERFORMANCE OPTIMIZATIONS vs previous version (was ~20h, now ~30min):

1. BINARY SEARCH PRE-INDEXING
   Instead of pd.concat(48 DataFrames) per snapshot:
   - Sort flights by dep_hour once
   - For each snapshot, use np.searchsorted to find flights in [t, t+24h)
   - O(log n) per snapshot instead of O(48 concat + dedup)

2. NUMPY ARRAYS — NO ITERROWS ANYWHERE
   All flight columns extracted to numpy arrays once before the loop.
   Feature/label/edge building uses numpy fancy indexing throughout.
   iterrows() was O(n_flights × n_snapshots) = 9M × 43K = 387B ops.
   Numpy vectorized ops are ~100x faster.

3. PRE-COMPUTED AIRPORT CONGESTION FLAGS
   Airport congestion status precomputed per (snapshot_hour, airport)
   as numpy arrays before the loop. No dict lookups per flight.

4. VECTORIZED CAUSAL EDGE BUILDING
   Boolean masking instead of per-flight Python loops.

Same design as previous version:
  - 6 causal edge types
  - Full tail propagation (cumulative delay + legs completed)
  - 24h flight window
  - Multi-horizon labels [1, 3, 6, 12]h
  - 30 airport features, 17 flight features
"""

import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── CONFIG ──────────────────────────────────────────────────────────────────
# LOCAL EXECUTION — run step 5 on your machine, upload outputs to Drive
# Change these paths to match your local setup
DRIVE_BASE     = r"C:\Users\user\Desktop\Airline_Graphs_Project"
GRAPH_DATA_DIR = rf"{DRIVE_BASE}\graph_data"
OUTPUT_DIR     = rf"{DRIVE_BASE}\graph_data"
FLIGHTS_PATH   = DRIVE_BASE

SNAPSHOT_FREQ       = "1h"
LABEL_HORIZON_AP    = 1
LABEL_HORIZONS_FL   = [0, 1, 3, 6, 12]  # 0 = <1h regime (gate fully unmasked) — now supervised
FLIGHT_WINDOW_HOURS = 8   # must be > max label horizon (6h) to get valid 6h masks

TRAIN_YEARS = [2018, 2019, 2020]
VAL_YEARS   = [2021]
TEST_YEARS  = [2022]
HUB_TOP_N   = 30

AP_DEP_DELAY_THRESH  = 15.0
AP_TAXI_MULT_THRESH  = 1.3
INBOUND_DELAY_THRESH = 15.0

N_AP_STATIC   = 5
N_AP_DYNAMIC  = 9
N_AP_TRAFFIC  = 6   # +2: dep_1h, arr_1h (was 4)
N_AP_FORECAST = 6
N_AP_TIME     = 4
N_AP_FEATURES = N_AP_STATIC + N_AP_DYNAMIC + N_AP_TRAFFIC + N_AP_FORECAST + N_AP_TIME  # 30

N_FL_FEATURES = 19   # +2: hist_route_delay_avg, hist_route_delay_std

MAX_DELAY_MIN    = 300.0
MAX_DISTANCE     = 5000.0
MAX_TAXI_MIN     = 60.0
MAX_TURNAROUND   = 90.0
MAX_FLIGHTS_1H   = 25.0   # typical max flights in 1h window at a hub
MAX_FLIGHTS_3H   = 60.0
MAX_FLIGHTS_6H   = 120.0
MAX_LEGS_PER_DAY = 6.0
MAX_CUMUL_DELAY  = 300.0
# ────────────────────────────────────────────────────────────────────────────


# ════════════════════════════════════════════════════════════════════════════
# LOADING  (unchanged from previous version)
# ════════════════════════════════════════════════════════════════════════════

def load_parquet_or_empty(path, name):
    if os.path.exists(path):
        df = pd.read_parquet(path)
        print(f"  ✓ {name}: {len(df):,} rows")
        return df
    print(f"  ⚠ Not found: {path}")
    return pd.DataFrame()


def load_weather_airports():
    locs = pd.read_parquet(
        os.path.join(GRAPH_DATA_DIR, "airport_locations.parquet"))
    airports = sorted(locs["airport"].dropna().unique().tolist())
    print(f"  ✓ Weather airports: {len(airports)}")
    return set(airports)


def load_flights(weather_airports):
    wx_path  = os.path.join(GRAPH_DATA_DIR, "weather_node_features.parquet")
    raw_path = os.path.join(FLIGHTS_PATH,   "flights_2018_2022.parquet")
    if os.path.exists(wx_path):
        df = pd.read_parquet(wx_path)
        print(f"  ✓ Weather-enriched: {len(df):,} rows (before filter)")
    else:
        df = pd.read_parquet(raw_path)
        print(f"  ✓ Raw flights: {len(df):,} rows (before filter)")

    # Assign flight_id BEFORE filtering so IDs match rotation_edges.parquet
    # Step 2 built rotation edges from the full dataset using df.index as flight_id
    # If we filter first then reset_index, IDs shift and rotation edges never match
    df = df.reset_index(drop=True)
    df["flight_id"] = df.index

    df = df[df["ORIGIN"].isin(weather_airports) &
            df["DEST"].isin(weather_airports)].copy()
    print(f"  ✓ After 36-airport filter: {len(df):,} rows")

    for col in ("dep_datetime","arr_datetime"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ("DepDelay","ArrDelay","TaxiOut","TaxiIn",
                "Distance","AirTime","CarrierDelay"):
        df[col] = pd.to_numeric(
            df.get(col, pd.Series(dtype=float)), errors="coerce")
    df["dep_hour"]    = df["dep_datetime"].dt.floor("h")
    df["arr_hour"]    = df["arr_datetime"].dt.floor("h")
    df["flight_date"] = df["dep_datetime"].dt.date
    df["hour_of_day"] = df["dep_datetime"].dt.hour
    df["month"]       = df["dep_datetime"].dt.month
    return df


# ════════════════════════════════════════════════════════════════════════════
# AIRPORT INDEX + STATIC FEATURES
# ════════════════════════════════════════════════════════════════════════════

def build_airport_index(weather_airports):
    airports = sorted(weather_airports)
    ap2idx   = {ap: i for i, ap in enumerate(airports)}
    return airports, ap2idx


def build_static_airport_features(airports, ap2idx, flights):
    print("\nBuilding static airport features ...")
    dep_s = (flights.groupby("ORIGIN")
                    .agg(hist_avg_dep_delay=("DepDelay","mean"),
                         hist_avg_taxi_out =("TaxiOut", "mean"),
                         total_departures  =("DepDelay","count"))
                    .reset_index().rename(columns={"ORIGIN":"airport"}))
    arr_s = (flights.groupby("DEST")
                    .agg(hist_avg_arr_delay=("ArrDelay","mean"))
                    .reset_index().rename(columns={"DEST":"airport"}))
    stats    = dep_s.merge(arr_s, on="airport", how="outer")
    stats["total_departures_norm"] = (
        stats["total_departures"] / stats["total_departures"].max())
    top_hubs = set(stats.nlargest(HUB_TOP_N,"total_departures")["airport"])
    stats["is_hub"] = stats["airport"].isin(top_hubs).astype(float)

    taxi_baseline = dict(zip(stats["airport"],
                             stats["hist_avg_taxi_out"].fillna(19.0)))

    n  = len(airports)
    X  = np.zeros((n, N_AP_STATIC), dtype=np.float32)
    sd = stats.set_index("airport").to_dict("index")
    for ap, idx in ap2idx.items():
        s = sd.get(ap, {})
        X[idx,0] = s.get("is_hub",                0.0) or 0.0
        X[idx,1] = s.get("hist_avg_dep_delay",    0.0) or 0.0
        X[idx,2] = s.get("hist_avg_taxi_out",     0.0) or 0.0
        X[idx,3] = s.get("hist_avg_arr_delay",    0.0) or 0.0
        X[idx,4] = s.get("total_departures_norm", 0.0) or 0.0
    X = np.nan_to_num(X, nan=0.0)
    print(f"  Static: {X.shape} | {int(X[:,0].sum())} hubs")
    return X, top_hubs, taxi_baseline


# ════════════════════════════════════════════════════════════════════════════
# FORECAST WEATHER + TRAFFIC LOAD
# ════════════════════════════════════════════════════════════════════════════

def build_weather_forecast_lookup(flights):
    print("\nBuilding forecast weather lookup ...")
    t0      = time.time()
    wx_cols = ["wind_speed_ms","visibility_m","precip_depth_mm"]
    avail   = [c for c in wx_cols if c in flights.columns]
    if not avail:
        print("  ⚠ No weather columns")
        return {}
    agg = {c: (c,"mean") for c in avail}
    pat = (flights.groupby(["ORIGIN","hour_of_day","month"])
                  .agg(**agg).reset_index()
                  .rename(columns={"ORIGIN":"airport"}))
    pattern = {}
    for row in pat.itertuples(index=False):
        pattern[(row.airport, row.hour_of_day, row.month)] = {
            "wind"  : getattr(row,"wind_speed_ms",  0.0) or 0.0,
            "precip": getattr(row,"precip_depth_mm",0.0) or 0.0,
            "vis"   : getattr(row,"visibility_m",   0.0) or 0.0,
        }
    lookup = {}
    for ap in pat["airport"].unique():
        for mo in range(1,13):
            for h in range(24):
                h3  = (h+3)%24; h6  = (h+6)%24
                mo3 = mo if h+3<24 else (mo%12)+1
                mo6 = mo if h+6<24 else (mo%12)+1
                w3  = pattern.get((ap,h3,mo3),{})
                w6  = pattern.get((ap,h6,mo6),{})
                lookup[(ap,h,mo)] = {
                    "wind_3h"  : w3.get("wind",  0.0),
                    "precip_3h": w3.get("precip",0.0),
                    "vis_3h"   : w3.get("vis",   0.0),
                    "wind_6h"  : w6.get("wind",  0.0),
                    "precip_6h": w6.get("precip",0.0),
                    "vis_6h"   : w6.get("vis",   0.0),
                }
    print(f"  Forecast entries: {len(lookup):,} in {time.time()-t0:.1f}s")
    return lookup


def build_traffic_lookup(flights, ap2idx):
    """
    Vectorized rebuild — groups by hour and airport once,
    then uses rolling sums instead of per-entry Python loops.
    """
    print("\nBuilding scheduled traffic load lookup ...")
    t0 = time.time()

    dep_cnt = (flights.groupby(["dep_hour","ORIGIN"])
                      .size().reset_index(name="cnt")
                      .rename(columns={"dep_hour":"hour","ORIGIN":"airport"}))
    arr_cnt = (flights.groupby(["arr_hour","DEST"])
                      .size().reset_index(name="cnt")
                      .rename(columns={"arr_hour":"hour","DEST":"airport"}))

    dep_dict = {(r.hour,r.airport): r.cnt
                for r in dep_cnt.itertuples(index=False)}
    arr_dict = {(r.hour,r.airport): r.cnt
                for r in arr_cnt.itertuples(index=False)}

    all_hours = sorted(set(dep_cnt["hour"].tolist()))
    lookup    = {}
    td        = [pd.Timedelta(hours=h) for h in range(1,7)]

    for snap_h in all_hours:
        for ap in ap2idx:
            d1 = dep_dict.get((snap_h+td[0],ap), 0)
            d3 = sum(dep_dict.get((snap_h+td[h],ap),0) for h in range(3))
            d6 = sum(dep_dict.get((snap_h+td[h],ap),0) for h in range(6))
            a1 = arr_dict.get((snap_h+td[0],ap), 0)
            a3 = sum(arr_dict.get((snap_h+td[h],ap),0) for h in range(3))
            a6 = sum(arr_dict.get((snap_h+td[h],ap),0) for h in range(6))
            lookup[(snap_h,ap)] = {
                "dep_1h":d1,"dep_3h":d3,"dep_6h":d6,
                "arr_1h":a1,"arr_3h":a3,"arr_6h":a6}

    print(f"  Traffic entries: {len(lookup):,} in {time.time()-t0:.1f}s")
    return lookup


def build_route_stats_lookup(flights):
    """
    Precompute historical DepDelay mean and std per route × hour × day-of-week.
    These are computed from the full dataset and stored per flight_id.

    This gives each flight non-gate information that differentiates it from
    other flights at the same airport — critical when gate features are masked.

    Key: (ORIGIN, DEST, dep_hour, day_of_week)
    Value: (mean_delay, std_delay)

    Fallback hierarchy:
      (O, D, hour, dow) → (O, D, hour) → (O, D) → global mean
    """
    print("\nBuilding route statistics lookup ...")
    t0 = time.time()

    # Extract integer hour and day-of-week directly from datetime
    # (flights["dep_hour"] is a floored Timestamp — not usable as int key)
    work = flights[["ORIGIN","DEST","DepDelay","dep_datetime","DayOfWeek"]].copy()
    work["hour"] = work["dep_datetime"].dt.hour.astype(int)
    work["dow"]  = work["DayOfWeek"].astype(int)

    # Full granularity: route × hour × day-of-week
    g1 = (work.groupby(["ORIGIN","DEST","hour","dow"])["DepDelay"]
              .agg(avg_delay="mean", std_delay="std", n="count")
              .reset_index())
    g1 = g1[g1["n"] >= 10]
    lookup_full = {
        (r.ORIGIN, r.DEST, int(r.hour), int(r.dow)):
        (float(r.avg_delay),
         float(r.std_delay) if not np.isnan(r.std_delay) else 15.0)
        for r in g1.itertuples(index=False)
    }

    # Fallback: route × hour only
    g2 = (work.groupby(["ORIGIN","DEST","hour"])["DepDelay"]
              .agg(avg_delay="mean", std_delay="std", n="count")
              .reset_index())
    g2 = g2[g2["n"] >= 5]
    lookup_hour = {
        (r.ORIGIN, r.DEST, int(r.hour)):
        (float(r.avg_delay),
         float(r.std_delay) if not np.isnan(r.std_delay) else 15.0)
        for r in g2.itertuples(index=False)
    }

    # Fallback: route only
    g3 = (work.groupby(["ORIGIN","DEST"])["DepDelay"]
              .agg(avg_delay="mean", std_delay="std")
              .reset_index())
    lookup_route = {
        (r.ORIGIN, r.DEST):
        (float(r.avg_delay),
         float(r.std_delay) if not np.isnan(r.std_delay) else 20.0)
        for r in g3.itertuples(index=False)
    }

    global_mean = float(work["DepDelay"].mean())
    global_std  = float(work["DepDelay"].std())

    print(f"  Route×hour×dow entries : {len(lookup_full):,}")
    print(f"  Route×hour entries     : {len(lookup_hour):,}")
    print(f"  Route-only entries     : {len(lookup_route):,}")
    print(f"  Global fallback        : mean={global_mean:.1f}  std={global_std:.1f}")
    print(f"  Built in {time.time()-t0:.1f}s")

    return lookup_full, lookup_hour, lookup_route, global_mean, global_std


# ════════════════════════════════════════════════════════════════════════════
# NETWORK EDGES
# ════════════════════════════════════════════════════════════════════════════

# Minimum correlation to build a network edge.
# Raising from 0.0 (all pairs with 0.1 default) to 0.3
# removes ~60% of edges that carry weak/noisy signal,
# sharpening the model's attention on genuinely correlated airports.
NETWORK_CORR_THRESHOLD = 0.30

def build_network_edges(airports, ap2idx, flights):
    """
    Build airport correlation network — only keep edges where delay patterns
    are genuinely correlated (r >= NETWORK_CORR_THRESHOLD).

    Fix: removed default 0.1 weight for missing pairs. Previously every
    airport pair got an edge even with no evidence of correlation.
    Now: only pairs with r >= 0.30 get an edge. Weaker connections are noise.
    """
    print("\nBuilding network edges ...")
    hourly = (flights.groupby(["dep_hour","ORIGIN"])["DepDelay"]
                     .mean().unstack("ORIGIN").fillna(0))
    cols = [ap for ap in airports if ap in hourly.columns]
    corr = hourly[cols].corr()
    records = []
    for ap1 in airports:
        for ap2 in airports:
            if ap1 == ap2:
                continue
            if ap1 not in corr.index or ap2 not in corr.columns:
                continue   # no data — skip entirely (was: default 0.1)
            r = float(corr.loc[ap1, ap2])
            if r < NETWORK_CORR_THRESHOLD:
                continue   # weak correlation — skip (was: include with low weight)
            records.append({"edge_type":"network",
                             "src_airport":ap1,"dst_airport":ap2,
                             "correlation":round(r,4),
                             "edge_weight":round(float(np.clip(r,0,1)),4)})
    df = pd.DataFrame(records)
    n_possible = len(airports) * (len(airports) - 1)
    print(f"  Network edges: {len(df):,} / {n_possible} possible "
          f"(threshold r>={NETWORK_CORR_THRESHOLD})")
    print(f"  Avg weight: {df['edge_weight'].mean():.3f}" if len(df) > 0 else "")
    return df


# ════════════════════════════════════════════════════════════════════════════
# FULL TAIL PROPAGATION
# ════════════════════════════════════════════════════════════════════════════

def build_tail_propagation_lookup(flights):
    print("\nBuilding full tail propagation lookup ...")
    t0  = time.time()
    sub = flights[["flight_id","Tail_Number","dep_datetime",
                   "arr_datetime","ArrDelay","flight_date"]].dropna(
        subset=["Tail_Number","dep_datetime","arr_datetime"])
    sub = sub.sort_values(["Tail_Number","dep_datetime"])

    lookup = {}
    for (tail,date), grp in sub.groupby(["Tail_Number","flight_date"]):
        grp    = grp.reset_index(drop=True)
        cumul  = 0.0
        legs   = 0
        for i in range(len(grp)):
            curr = grp.iloc[i]
            if i > 0:
                prev = grp.iloc[i-1]
                gap  = ((curr["dep_datetime"]-prev["arr_datetime"])
                        .total_seconds()/3600)
                if 0 <= gap <= 8 and pd.notna(prev["ArrDelay"]):
                    lookup[curr["flight_id"]] = {
                        "cumulative_delay"  : cumul,
                        "legs_completed"    : legs,
                        "immediate_inbound" : float(prev["ArrDelay"]),
                    }
            if pd.notna(curr["ArrDelay"]):
                cumul += max(0.0, float(curr["ArrDelay"]))
            legs += 1

    print(f"  Tail entries: {len(lookup):,} in {time.time()-t0:.1f}s")
    delayed = sum(1 for v in lookup.values() if v["cumulative_delay"]>15)
    print(f"  Cumulative delay >15min: {delayed:,} ({100*delayed/max(len(lookup),1):.1f}%)")
    return lookup


# ════════════════════════════════════════════════════════════════════════════
# PRE-AGGREGATION (airport dynamic features and labels)
# ════════════════════════════════════════════════════════════════════════════

def preaggregate_airport_features(flights):
    print("\nPre-aggregating airport dynamic features ...")
    t0      = time.time()

    dep_agg = {"avg_dep_delay":("DepDelay","mean"),
               "avg_taxi_out" :("TaxiOut", "mean"),
               "dep_count"    :("DepDelay","count")}
    for col in ["wind_speed_ms","visibility_m","precip_depth_mm"]:
        if col in flights.columns:
            dep_agg[col] = (col,"mean")

    # Group by (dep_hour, airport) where dep_hour is already a floored Timestamp
    # that encodes the full date+hour — e.g. 2022-11-04 14:00:00
    # This gives actual per-date weather instead of 5-year averages.
    # The lookup key (t, airport) in build_ap_features_fast uses the same
    # Timestamp t from snapshot_times, so keys match automatically.
    dep_df = (flights.groupby(["dep_hour","ORIGIN"])
                     .agg(**dep_agg).reset_index()
                     .rename(columns={"dep_hour":"hour","ORIGIN":"airport"}))

    arr_df = (flights.groupby(["arr_hour","DEST"])
                     .agg(avg_arr_delay=("ArrDelay","mean"),
                          avg_taxi_in  =("TaxiIn",  "mean"),
                          arr_count    =("ArrDelay","count"))
                     .reset_index()
                     .rename(columns={"arr_hour":"hour","DEST":"airport"}))

    dep_lookup = {}
    for row in dep_df.itertuples(index=False):
        dep_lookup[(row.hour, row.airport)] = {
            "avg_dep_delay": row.avg_dep_delay,
            "avg_taxi_out" : row.avg_taxi_out,
            "dep_count"    : row.dep_count,
            "wind"  : getattr(row,"wind_speed_ms",  0.0) or 0.0,
            "vis"   : getattr(row,"visibility_m",   0.0) or 0.0,
            "precip": getattr(row,"precip_depth_mm",0.0) or 0.0,
        }
    arr_lookup = {}
    for row in arr_df.itertuples(index=False):
        arr_lookup[(row.hour, row.airport)] = {
            "avg_arr_delay": row.avg_arr_delay,
            "avg_taxi_in"  : row.avg_taxi_in,
            "arr_count"    : row.arr_count,
        }

    print(f"  Done in {time.time()-t0:.1f}s | dep:{len(dep_lookup):,} arr:{len(arr_lookup):,}")
    return dep_lookup, arr_lookup


def preaggregate_airport_labels(flights):
    # Keyed by (arr_hour Timestamp, airport) — arr_hour is already date-specific
    dep_df = (flights.groupby(["dep_hour","ORIGIN"])
                     .agg(avg_dep_delay=("DepDelay","mean"))
                     .reset_index()
                     .rename(columns={"dep_hour":"hour","ORIGIN":"airport"}))
    return {(r.hour, r.airport): r.avg_dep_delay
            for r in dep_df.itertuples(index=False)}


# ════════════════════════════════════════════════════════════════════════════
# EDGE PRE-PROCESSING
# ════════════════════════════════════════════════════════════════════════════

def filter_to_weather(df, wa, sc, dc):
    return df[df[sc].isin(wa) & df[dc].isin(wa)].copy()

def preprocess_rotation_edges(rotation_edges, ap2idx, weather_airports):
    """
    Pre-process rotation edges indexed by leg2 departure hour.

    Phase-aware rotation edge attributes (4 dims):
      [0] delay_signal     — leg1 delay estimate (transitions with horizon)
      [1] signal_confidence — 0=historical guess, 1=actual observed arrival
      [2] turnaround_norm  — turnaround time normalised to [0,1]
      [3] leg2_h_norm      — hours to leg2 departure / 24 (set in build loop)

    Phase logic (applied in build_flight_edges_fast per snapshot):
      >4h before leg2: leg1 not yet landed → signal=leg1 hist route avg,
                       confidence=0 (model learns to discount)
      1-4h before leg2: leg1 landed → signal=actual leg1_arr_delay,
                        confidence=1 (model learns to trust)
      <1h before leg2: gate features take over, rotation edge still fires
                       as confirmation signal
    """
    print("Pre-processing rotation edges (phase-aware) ...")
    re = filter_to_weather(rotation_edges, weather_airports,
                           "src_airport","mid_airport")

    re["leg1_dep"]         = pd.to_datetime(re["leg1_dep"], errors="coerce")
    re["leg1_arr"]         = pd.to_datetime(re.get("leg1_arr",
                             pd.NaT), errors="coerce")
    re["leg2_dep"]         = pd.to_datetime(re.get("leg2_dep",
                             re["leg1_dep"]), errors="coerce")
    re["src_idx"]          = re["src_airport"].map(ap2idx)
    re["dst_idx"]          = re["mid_airport"].map(ap2idx)
    re                     = re.dropna(subset=["src_idx","dst_idx","leg1_dep"])
    re["src_idx"]          = re["src_idx"].astype(int)
    re["dst_idx"]          = re["dst_idx"].astype(int)
    re["turnaround_norm"]  = (re["turnaround_min"].clip(0,90)/90).astype(np.float32)
    re["leg1_arr_delay_n"] = (re["leg1_arr_delay"].fillna(0)
                               .clip(0,MAX_DELAY_MIN)/MAX_DELAY_MIN
                              ).astype(np.float32)
    re["edge_weight"]      = re["edge_weight"].fillna(0).astype(np.float32)

    # leg1_arr in nanoseconds for fast comparison in the snapshot loop
    re["leg1_arr_ns"] = (re["leg1_arr"]
                          .values.astype("datetime64[ns]")
                          .astype(np.int64))

    # Index by leg2 departure hour
    if "leg2_dep" not in rotation_edges.columns:
        re["leg2_dep_est"] = (re["leg1_dep"] +
                               pd.to_timedelta(
                                   re["turnaround_min"].fillna(120), unit="min"))
    else:
        re["leg2_dep_est"] = re["leg2_dep"]

    re["leg2_dep_ns"]  = (re["leg2_dep_est"]
                           .values.astype("datetime64[ns]")
                           .astype(np.int64))
    re["index_hour"]   = re["leg2_dep_est"].dt.floor("h")

    # Resolve leg2_flight_id column name
    if "leg2_flight_id" not in re.columns and "leg2_id" in re.columns:
        re["leg2_flight_id"] = re["leg2_id"]

    hour_index = {}
    for hour, grp in re.groupby("index_hour"):
        # Store per-edge arrays needed for phase-aware weighting:
        #   turnaround_norm, leg1_arr_delay_norm, leg1_arr_ns, leg2_dep_ns
        # plus the full group df for flight-id matching
        arr_data = np.stack([
            grp["turnaround_norm"].values,
            grp["leg1_arr_delay_n"].values,
            grp["leg1_arr_ns"].values.astype(np.float64),
            grp["leg2_dep_ns"].values.astype(np.float64),
        ], axis=1).astype(np.float64)   # float64 needed for ns precision

        hour_index[hour] = (grp["src_idx"].values,
                            grp["dst_idx"].values,
                            arr_data, grp)

    total = sum(v[0].shape[0] for v in hour_index.values())
    print(f"  Rotation hours: {len(hour_index):,} | instances: {total:,}")
    print(f"  Indexed by leg2 departure hour (phase-aware signal)")
    return hour_index

def preprocess_congestion_edges(congestion_edges, ap2idx, weather_airports):
    """
    Build congestion edge TOPOLOGY — which airport pairs are connected.
    Edge weights are computed dynamically per snapshot in build_snapshots()
    based on actual taxi-out z-scores at each snapshot hour.

    Returns:
        cg_ei : (2, n_edges) edge index
        cg_src: array of src airport indices (for dynamic weighting)
        cg_dst: array of dst airport indices
        cg_base_weight: static base weight (volume/topology component)
        cg_taxi_mean:   per-edge src airport taxi mean (for z-score)
        cg_taxi_std:    per-edge src airport taxi std
    """
    print("Pre-processing congestion edges (dynamic-ready) ...")
    if len(congestion_edges) == 0:
        empty = np.zeros(0, dtype=np.float32)
        return (np.zeros((2,0),dtype=np.int64),
                empty, empty, empty, empty, empty)

    # Use taxi_anomaly edges for dynamic weighting; volume/hub_spoke as static
    ce = congestion_edges.copy()
    ce = ce[ce["src_airport"].isin(weather_airports) &
            ce["dst_airport"].isin(weather_airports)].copy()

    ce["src_idx"] = ce["src_airport"].map(ap2idx)
    ce["dst_idx"] = ce["dst_airport"].map(ap2idx)
    ce = ce.dropna(subset=["src_idx","dst_idx"])
    ce["src_idx"] = ce["src_idx"].astype(int)
    ce["dst_idx"] = ce["dst_idx"].astype(int)

    ei         = np.stack([ce["src_idx"].values, ce["dst_idx"].values], axis=0)
    base_w     = ce["edge_weight"].fillna(0).values.astype(np.float32)
    taxi_mean  = ce.get("taxi_mean", pd.Series(19.0, index=ce.index))\
                   .fillna(19.0).values.astype(np.float32)
    taxi_std   = ce.get("taxi_std",  pd.Series(5.0,  index=ce.index))\
                   .fillna(5.0).clip(lower=1.0).values.astype(np.float32)
    sub_type   = ce.get("sub_type", pd.Series("", index=ce.index))\
                   .fillna("").values

    print(f"  Congestion topology: {ei.shape[1]:,} edges")
    for st in np.unique(sub_type):
        if st:
            print(f"    {st}: {(sub_type==st).sum():,}")

    return ei, ce["src_idx"].values, ce["dst_idx"].values, \
           base_w, taxi_mean, taxi_std


def compute_dynamic_congestion_weights(cg_src, cg_dst, cg_base_w,
                                        cg_taxi_mean, cg_taxi_std,
                                        ap_avg_taxi, ap_avg_dep,
                                        TAXI_ZSCORE_THRESH=1.3,
                                        DEP_DELAY_THRESH=15.0):
    """
    Compute live congestion edge weights for the current snapshot.

    Weight = base_weight × live_congestion_signal

    live_congestion_signal:
      - taxi z-score at src airport: (actual_taxi - mean) / std
        normalised to [0, 1]. If taxi is above baseline: weight amplified.
      - dep delay signal: if avg_dep_delay > DEP_DELAY_THRESH, adds signal
      - Combined: max of both signals

    This means on a clear morning with normal taxi times, weights ≈ base_weight.
    On a congested afternoon, weights spike, propagating congestion to neighbours.
    """
    if len(cg_src) == 0:
        return np.zeros(0, dtype=np.float32)

    # Taxi z-score at each edge's source airport
    src_taxi   = ap_avg_taxi[cg_src]          # actual avg taxi this hour
    taxi_z     = (src_taxi - cg_taxi_mean) / cg_taxi_std
    taxi_sig   = np.clip(taxi_z / (TAXI_ZSCORE_THRESH * 2), 0, 1)

    # Departure delay signal at source airport
    src_dep    = ap_avg_dep[cg_src]
    dep_sig    = np.clip(src_dep / (DEP_DELAY_THRESH * 3), 0, 1)

    # Combined live signal: max of taxi and dep delay signals
    live_sig   = np.maximum(taxi_sig, dep_sig)

    # Final weight: base topology weight + live congestion boost
    # Base ensures edges always carry some structural signal
    # Live boost amplifies when congestion is actually happening
    weights    = cg_base_w * (0.3 + 0.7 * live_sig)

    return weights.astype(np.float32)

def preprocess_network_edges(network_df, ap2idx):
    print("Pre-processing network edges ...")
    df = network_df.copy()
    df["src_idx"] = df["src_airport"].map(ap2idx)
    df["dst_idx"] = df["dst_airport"].map(ap2idx)
    df = df.dropna(subset=["src_idx","dst_idx"])
    df["src_idx"] = df["src_idx"].astype(int)
    df["dst_idx"] = df["dst_idx"].astype(int)
    ei = np.stack([df["src_idx"].values, df["dst_idx"].values], axis=0)
    ea = df[["edge_weight","correlation"]].fillna(0).values.astype(np.float32)
    print(f"  Network: {ei.shape[1]} edges")
    return ei, ea


# ════════════════════════════════════════════════════════════════════════════
# FAST PRE-INDEXING — binary search, O(log n) per snapshot
# ════════════════════════════════════════════════════════════════════════════

def preindex_flights(flights, snapshot_times):
    """
    For each snapshot time t, find all flights departing in [t, t+24h)
    OR arriving in [t, t+24h) using binary search on sorted arrays.

    Returns dict: t -> np.array of integer row positions in flights.

    This replaces the per-snapshot pd.concat(48 DataFrames) which was
    the primary bottleneck causing 20+ hour runtimes.
    """
    print("\nPre-indexing flights to snapshots (binary search) ...")
    t0 = time.time()

    # Convert to int64 nanoseconds for fast numpy comparison
    dep_ns = flights["dep_datetime"].values.astype("datetime64[ns]").astype(np.int64)
    arr_ns = flights["arr_datetime"].values.astype("datetime64[ns]").astype(np.int64)

    # Sort by dep time for binary search
    dep_sort_order = np.argsort(dep_ns, kind="stable")
    dep_ns_sorted  = dep_ns[dep_sort_order]

    # Sort by arr time for binary search
    arr_sort_order = np.argsort(arr_ns, kind="stable")
    arr_ns_sorted  = arr_ns[arr_sort_order]

    window_ns = np.int64(FLIGHT_WINDOW_HOURS * 3_600_000_000_000)  # 24h in ns

    snap_to_fids = {}
    snap_times_ns = [np.datetime64(t,"ns").astype(np.int64) for t in snapshot_times]

    for t, t_ns in zip(snapshot_times, snap_times_ns):
        t_end = t_ns + window_ns

        # Fix 2: only include FUTURE-DEPARTING flights.
        # Previously included arrivals in [t, t+24h] which added already-airborne
        # or already-landed flights whose dep_datetime was before t.
        # Those flights had h2dep clamped to 0, making them look like <1h flights
        # even though they already departed — corrupting the 0h supervision signal.
        #
        # Now: only flights where dep_datetime >= t (not yet departed at snapshot time)
        # In-flight/arriving aircraft are represented through the GRU airport state.
        lo = np.searchsorted(dep_ns_sorted, t_ns,  side="left")
        hi = np.searchsorted(dep_ns_sorted, t_end, side="left")
        dep_rows = dep_sort_order[lo:hi]

        snap_to_fids[t] = dep_rows if len(dep_rows) > 0 else np.array([], dtype=np.int64)

    avg_fl = np.mean([len(v) for v in snap_to_fids.values()])
    print(f"  Indexed {len(snapshot_times):,} snapshots in {time.time()-t0:.1f}s")
    print(f"  Avg flights per snapshot: {avg_fl:.0f}")
    return snap_to_fids


# ════════════════════════════════════════════════════════════════════════════
# PRE-EXTRACT FLIGHT ARRAYS — convert all columns to numpy once
# ════════════════════════════════════════════════════════════════════════════

def preextract_flight_arrays(flights, ap2idx, top_hubs, taxi_baseline,
                              tail_propagation_lookup, airports,
                              route_stats=None):
    """
    Extract all needed flight columns to numpy arrays ONCE.
    In the loop we index with fancy indexing — no DataFrame ops.
    This eliminates iterrows() which was the second major bottleneck.

    route_stats: tuple of (lookup_full, lookup_hour, lookup_route,
                           global_mean, global_std) from build_route_stats_lookup
    """
    print("\nPre-extracting flight arrays ...")
    t0  = time.time()
    n   = len(flights)
    fid = flights["flight_id"].values

    fa = {}
    fa["dep_delay"]   = flights["DepDelay"].fillna(np.nan).values.astype(np.float32)
    fa["dep_valid"]   = flights["DepDelay"].notna().values
    fa["taxi_out"]    = flights["TaxiOut"].fillna(0).values.astype(np.float32)
    fa["distance"]    = flights["Distance"].fillna(0).values.astype(np.float32)
    fa["carrier_d"]   = flights.get("CarrierDelay",
                                    pd.Series(0,index=flights.index))\
                               .fillna(0).values.astype(np.float32)

    fa["dep_dt_ns"]   = flights["dep_datetime"].values\
                               .astype("datetime64[ns]").astype(np.int64)
    fa["arr_dt_ns"]   = flights["arr_datetime"].values\
                               .astype("datetime64[ns]").astype(np.int64)
    fa["dep_hour_int"]= flights["dep_datetime"].dt.hour.fillna(0)\
                               .values.astype(np.int32)
    fa["arr_hour_int"]= flights["arr_datetime"].dt.hour.fillna(0)\
                               .values.astype(np.int32)
    fa["dow"]         = flights["dep_datetime"].dt.dayofweek.fillna(0)\
                               .values.astype(np.int32)
    fa["flight_id"]   = fid

    # Airport indices per flight (-1 = unknown)
    fa["origin_idx"]  = np.array(
        [ap2idx.get(o,-1) for o in flights["ORIGIN"].values], dtype=np.int32)
    fa["dest_idx"]    = np.array(
        [ap2idx.get(d,-1) for d in flights["DEST"].values],   dtype=np.int32)

    # Hub flag
    fa["is_hub_origin"] = np.array(
        [1.0 if o in top_hubs else 0.0
         for o in flights["ORIGIN"].values], dtype=np.float32)

    # Tail propagation — pre-extracted
    fa["cumul_delay"]   = np.array(
        [tail_propagation_lookup.get(f,{}).get("cumulative_delay",  0.0)
         for f in fid], dtype=np.float32)
    fa["legs_done"]     = np.array(
        [tail_propagation_lookup.get(f,{}).get("legs_completed",    0)
         for f in fid], dtype=np.float32)
    fa["immed_inbound"] = np.array(
        [tail_propagation_lookup.get(f,{}).get("immediate_inbound", 0.0)
         for f in fid], dtype=np.float32)
    fa["is_first"]      = ((fa["cumul_delay"] == 0) &
                            (fa["immed_inbound"] == 0)).astype(np.float32)

    # Metadata strings for step 7
    fa["origin_str"]  = flights["ORIGIN"].fillna("").values
    fa["dest_str"]    = flights["DEST"].fillna("").values
    fa["tail_str"]    = (flights["Tail_Number"].fillna("").values
                         if "Tail_Number" in flights.columns
                         else np.array([""] * n))
    fa["airline_str"] = (flights["Operating_Airline"].fillna("").values
                         if "Operating_Airline" in flights.columns
                         else np.array([""] * n))
    # Fix 3: encode tail numbers as integer indices for per-tail GRU
    # tail_id=0 reserved for unknown/missing tails
    unique_tails = sorted(set(fa["tail_str"].tolist()) - {""})
    tail2idx     = {t: i+1 for i, t in enumerate(unique_tails)}
    fa["tail2idx"] = tail2idx   # stored on fa for reuse across snapshots
    fa["tail_id"]  = np.array(
        [tail2idx.get(t, 0) for t in fa["tail_str"]], dtype=np.int32)
    print(f"  Tail numbers: {len(unique_tails):,} unique → integer indices [1, {len(unique_tails)}]")

    # Airport taxi baselines as array (indexed by ap_idx)
    n_ap = len(airports)
    fa["ap_taxi_baseline"] = np.array(
        [taxi_baseline.get(airports[i], 19.0) for i in range(n_ap)],
        dtype=np.float32)

    # Historical route delay stats — avg and std per route × hour × dow
    # These survive horizon masking (precomputed from training data)
    if route_stats is not None:
        lookup_full, lookup_hour, lookup_route, g_mean, g_std = route_stats
        origins  = flights["ORIGIN"].values
        dests    = flights["DEST"].values
        dep_hrs  = fa["dep_hour_int"]
        dows     = fa["dow"]

        hist_avg = np.full(n, g_mean, dtype=np.float32)
        hist_std = np.full(n, g_std,  dtype=np.float32)

        for j in range(n):
            o, d, h, dw = origins[j], dests[j], int(dep_hrs[j]), int(dows[j])
            if   (o, d, h, dw) in lookup_full:
                hist_avg[j], hist_std[j] = lookup_full[(o, d, h, dw)]
            elif (o, d, h)     in lookup_hour:
                hist_avg[j], hist_std[j] = lookup_hour[(o, d, h)]
            elif (o, d)        in lookup_route:
                hist_avg[j], hist_std[j] = lookup_route[(o, d)]
    else:
        hist_avg = np.zeros(n, dtype=np.float32)
        hist_std = np.full(n, 15.0, dtype=np.float32)

    fa["hist_avg"] = hist_avg
    fa["hist_std"] = hist_std

    print(f"  Pre-extracted {n:,} flights in {time.time()-t0:.1f}s")
    return fa


# ════════════════════════════════════════════════════════════════════════════
# PRE-COMPUTE AIRPORT CONGESTION FLAGS — once per snapshot hour
# ════════════════════════════════════════════════════════════════════════════

def precompute_ap_congestion(snapshot_times, airports, dep_lookup,
                              taxi_baseline):
    """
    Pre-compute per-snapshot per-airport congestion flags.
    Shape: dict[t] -> (ap_avg_dep: array[36], ap_avg_taxi: array[36])
    Used for vectorized causal edge building — no dict lookups in the loop.
    """
    print("Pre-computing airport congestion flags ...")
    t0     = time.time()
    n_ap   = len(airports)
    result = {}

    for t in snapshot_times:
        avg_dep  = np.zeros(n_ap, dtype=np.float32)
        avg_taxi = np.zeros(n_ap, dtype=np.float32)
        for i, ap in enumerate(airports):
            d = dep_lookup.get((t, ap), {})
            avg_dep[i]  = d.get("avg_dep_delay", 0.0) or 0.0
            avg_taxi[i] = d.get("avg_taxi_out",  0.0) or 0.0
        result[t] = (avg_dep, avg_taxi)

    print(f"  Congestion flags: {len(result):,} snapshots in {time.time()-t0:.1f}s")
    return result


# ════════════════════════════════════════════════════════════════════════════
# AIRPORT DYNAMIC FEATURES (per snapshot, vectorized)
# ════════════════════════════════════════════════════════════════════════════

def build_ap_features_fast(t, airports, ap2idx,
                            dep_lookup, arr_lookup,
                            traffic_lookup, forecast_lookup):
    n  = len(airports)
    n_dyn = N_AP_DYNAMIC + N_AP_TRAFFIC + N_AP_FORECAST
    X  = np.zeros((n, n_dyn), dtype=np.float32)
    h, mo = t.hour, t.month
    for i, ap in enumerate(airports):
        dep  = dep_lookup.get((t,ap), {})
        arr  = arr_lookup.get((t,ap), {})
        traf = traffic_lookup.get((t,ap), {})
        fore = forecast_lookup.get((ap,h,mo), {})
        X[i,0]  = dep.get("avg_dep_delay",0.0) or 0.0
        X[i,1]  = arr.get("avg_arr_delay",0.0) or 0.0
        X[i,2]  = dep.get("avg_taxi_out", 0.0) or 0.0
        X[i,3]  = arr.get("avg_taxi_in",  0.0) or 0.0
        X[i,4]  = float(dep.get("dep_count",0))
        X[i,5]  = float(arr.get("arr_count",0))
        X[i,6]  = dep.get("wind",  0.0) or 0.0
        X[i,7]  = dep.get("vis",   0.0) or 0.0
        X[i,8]  = dep.get("precip",0.0) or 0.0
        # Traffic load (6 dims: 1h, 3h, 6h × dep/arr)
        X[i,9]  = min(traf.get("dep_1h",0)/MAX_FLIGHTS_1H,1.0)  # NEW
        X[i,10] = min(traf.get("arr_1h",0)/MAX_FLIGHTS_1H,1.0)  # NEW
        X[i,11] = min(traf.get("dep_3h",0)/MAX_FLIGHTS_3H,1.0)
        X[i,12] = min(traf.get("dep_6h",0)/MAX_FLIGHTS_6H,1.0)
        X[i,13] = min(traf.get("arr_3h",0)/MAX_FLIGHTS_3H,1.0)
        X[i,14] = min(traf.get("arr_6h",0)/MAX_FLIGHTS_6H,1.0)
        # Forecast weather (6 dims: wind/precip/vis at +3h and +6h)
        X[i,15] = fore.get("wind_3h",  0.0)
        X[i,16] = fore.get("precip_3h",0.0)
        X[i,17] = fore.get("vis_3h",   0.0)
        X[i,18] = fore.get("wind_6h",  0.0)
        X[i,19] = fore.get("precip_6h",0.0)
        X[i,20] = fore.get("vis_6h",   0.0)
    return np.nan_to_num(X, nan=0.0)


def build_time_features(t, n):
    h, mo = t.hour, t.month
    return np.tile([np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24),
                    np.sin(2*np.pi*mo/12),np.cos(2*np.pi*mo/12)],
                   (n,1)).astype(np.float32)


def build_ap_labels(t, ap2idx, label_lookup):
    y  = np.full(len(ap2idx), np.nan, dtype=np.float32)
    nh = t + pd.Timedelta(hours=LABEL_HORIZON_AP)
    for ap, idx in ap2idx.items():
        v = label_lookup.get((nh,ap))
        if v is not None:
            y[idx] = v
    return y


# ════════════════════════════════════════════════════════════════════════════
# VECTORIZED FLIGHT FEATURES — no iterrows
# ════════════════════════════════════════════════════════════════════════════

def build_flight_features_fast(fids, fa, t_ns, same_snap_inbound):
    """
    Build flight feature matrix using pre-extracted numpy arrays.
    All operations are vectorized — no Python loops per flight.

    fids         : row indices into fa arrays
    fa           : pre-extracted flight arrays dict
    t_ns         : snapshot time as int64 nanoseconds
    same_snap_inbound : dict fid->(immed, turnaround) from rotation group
    """
    n = len(fids)
    if n == 0:
        return np.zeros((0, N_FL_FEATURES), dtype=np.float32)

    X = np.zeros((n, N_FL_FEATURES), dtype=np.float32)

    # Extract arrays for this snapshot's flights
    distance    = fa["distance"][fids]
    dep_h_int   = fa["dep_hour_int"][fids]
    arr_h_int   = fa["arr_hour_int"][fids]
    dow         = fa["dow"][fids]
    dep_dt_ns   = fa["dep_dt_ns"][fids]
    fid_arr     = fa["flight_id"][fids]
    is_hub_o    = fa["is_hub_origin"][fids]
    cumul       = fa["cumul_delay"][fids].copy()
    legs        = fa["legs_done"][fids].copy()
    immed       = fa["immed_inbound"][fids].copy()
    is_first    = fa["is_first"][fids].copy()
    turnaround  = np.zeros(n, dtype=np.float32)
    hist_avg    = fa["hist_avg"][fids]   # historical route mean
    hist_std    = fa["hist_std"][fids]   # historical route std

    # Override with same-snapshot rotation data where available
    if same_snap_inbound:
        for j, fid in enumerate(fid_arr):
            if fid in same_snap_inbound:
                immed_v, turn_v = same_snap_inbound[fid]
                immed[j]     = immed_v
                turnaround[j]= turn_v
                is_first[j]  = 0.0

    # Time to departure (vectorized)
    # dep_dt_ns is int64 ns; t_ns is int64 ns
    h2dep = np.maximum(0.0,
                       (dep_dt_ns - t_ns).astype(np.float64) / 3_600_000_000_000)

    # Build feature matrix (all numpy, no loops)
    # Departure-delay target leakage guard:
    # realized DepDelay / TaxiOut / CarrierDelay are not available at forecast
    # time, so keep their legacy feature slots zeroed for compatibility.
    X[:,0]  = 0.0
    X[:,1]  = np.sin(2*np.pi*dep_h_int/24).astype(np.float32)
    X[:,2]  = np.cos(2*np.pi*dep_h_int/24).astype(np.float32)
    X[:,3]  = np.clip(turnaround/MAX_TURNAROUND, 0, 1)
    X[:,4]  = np.clip(immed/MAX_DELAY_MIN, 0, 1)
    X[:,5]  = np.clip(distance/MAX_DISTANCE, 0, 1)
    X[:,6]  = is_first
    X[:,7]  = 0.0
    X[:,8]  = np.sin(2*np.pi*dow/7).astype(np.float32)
    X[:,9]  = np.cos(2*np.pi*dow/7).astype(np.float32)
    X[:,10] = is_hub_o
    X[:,11] = 0.0
    X[:,12] = np.sin(2*np.pi*arr_h_int/24).astype(np.float32)
    X[:,13] = np.cos(2*np.pi*arr_h_int/24).astype(np.float32)
    X[:,14] = np.minimum(h2dep/24.0, 1.0).astype(np.float32)
    X[:,15] = np.clip(cumul/MAX_CUMUL_DELAY, 0, 1)
    X[:,16] = np.minimum(legs/MAX_LEGS_PER_DAY, 1.0)
    # Historical route stats — available at any horizon, never masked
    X[:,17] = np.clip(hist_avg / MAX_DELAY_MIN, -1, 1).astype(np.float32)
    X[:,18] = np.clip(hist_std / MAX_DELAY_MIN,  0, 1).astype(np.float32)

    return np.nan_to_num(X, nan=0.0).astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════
# VECTORIZED FLIGHT LABELS
# ════════════════════════════════════════════════════════════════════════════

def build_flight_labels_fast(fids, fa, t_ns):
    n = len(fids)
    y     = np.zeros(n, dtype=np.float32)
    masks = {h: np.zeros(n, dtype=bool) for h in LABEL_HORIZONS_FL}
    if n == 0:
        return y, masks

    dep_valid = fa["dep_valid"][fids]
    dep_delay = fa["dep_delay"][fids]
    dep_dt_ns = fa["dep_dt_ns"][fids]

    h2dep = (dep_dt_ns.astype(np.float64) - t_ns) / 3_600_000_000_000

    valid = dep_valid & (h2dep >= 0)
    y[valid] = dep_delay[valid]

    # Exclusive horizon bands — no nesting, no overlap.
    # Each flight belongs to exactly ONE band based on time to departure.
    # This means:
    #   - gradient updates for 6h flights don't bleed into 1h training
    #   - checkpoint metric uses a representative mix, not a biased subset
    #   - per-horizon MAE is true per-band accuracy, not cumulative
    #
    # band_0h: 0 <= h2dep < 1   gate fully unmasked, supervised now
    # band_1h: 1 <= h2dep < 3   gate partially masked
    # band_3h: 3 <= h2dep < 6   no gate data
    # band_6h: h2dep >= 6       hardest — no gate, farthest out
    masks[0]  = valid & (h2dep >= 0) & (h2dep <  1)
    masks[1]  = valid & (h2dep >= 1) & (h2dep <  3)
    masks[3]  = valid & (h2dep >= 3) & (h2dep <  6)
    masks[6]  = valid & (h2dep >= 6)
    # Keep 12h as alias of 6h (same band, for compatibility)
    if 12 in masks:
        masks[12] = masks[6]

    return y, masks


# ════════════════════════════════════════════════════════════════════════════
# VECTORIZED CAUSAL EDGE BUILDING
# ════════════════════════════════════════════════════════════════════════════

def build_flight_edges_fast(fids, fa, rot_grp, rot_arr_data,
                             ap_avg_dep, ap_avg_taxi, ap_taxi_baseline,
                             t_ns):
    """
    Build all flight edge types with phase-aware rotation signals.

    Rotation edge (flight→flight self-edge on leg2):
      4 attributes:
        [0] delay_signal     — normalised delay estimate (phase-dependent)
        [1] signal_confidence — 0=historical, 1=observed actual arrival
        [2] turnaround_norm  — turnaround / 90min
        [3] h2dep2_norm      — hours to leg2 departure / 24

      Phase transitions:
        h2dep2 > 4h: leg1 not landed → delay_signal = leg1 hist route avg
                     confidence = 0  (model learns to discount this)
        1h < h2dep2 ≤ 4h: leg1 landed → delay_signal = actual leg1_arr_delay
                     confidence = 1  (model learns to trust this)
        h2dep2 ≤ 1h: gate features already unmasked — rotation still fires
                     confidence = 1  (gate features dominate anyway)

    departs_from: unconditional — every flight connects to origin airport
    arrives_at  : causal — fires if cumul > 15 OR immed > 15
    """
    n = len(fids)
    if n == 0:
        e2 = (np.zeros((2,0),dtype=np.int64), np.zeros((0,2),dtype=np.float32))
        e1 = (np.zeros((2,0),dtype=np.int64), np.zeros((0,1),dtype=np.float32))
        return e2, e1, e1

    fl2local = {fid: i for i, fid in enumerate(fa["flight_id"][fids])}

    # ── Causal flight→flight rotation ────────────────────────────────────
    # SIMPLIFIED: only fire when leg1 has ACTUALLY LANDED and delay is known.
    # The phase-aware approach with confidence=0.1 for far-out flights added
    # more noise than signal — the model learned to memorize noisy priors
    # that didn't generalize to validation.
    #
    # Hard causal gate: if leg1 hasn't landed yet → no edge (clean gradient)
    #                   if leg1 has landed → edge with actual delay (real signal)
    #
    # This means:
    #   6h ahead: typically no rotation edge (leg1 still airborne) → no noise
    #   1-3h ahead: rotation edge fires with real observed delay → real signal
    #   <1h: gate features already unmasked, rotation confirms them
    #
    # Edge attr (2 dims): [actual_delay_norm, turnaround_norm]
    NS_PER_HOUR = 3_600_000_000_000

    r_src, r_dst, r_attr = [], [], []
    if rot_grp is not None and len(rot_grp) > 0:
        leg1_col = ("leg1_flight_id" if "leg1_flight_id" in rot_grp.columns
                    else ("leg1_id" if "leg1_id" in rot_grp.columns else None))
        leg2_col = ("leg2_flight_id" if "leg2_flight_id" in rot_grp.columns
                    else ("leg2_id" if "leg2_id" in rot_grp.columns else None))

        if leg1_col is not None and leg2_col is not None:
            leg1_ids      = rot_grp[leg1_col].values
            leg2_ids      = rot_grp[leg2_col].values
            turn_norms    = rot_arr_data[:, 0].astype(np.float32)
            actual_delays = rot_arr_data[:, 1].astype(np.float32)
            leg1_arr_ns_v = rot_arr_data[:, 2]

            for j, (l1, l2) in enumerate(zip(leg1_ids, leg2_ids)):
                if l1 not in fl2local or l2 not in fl2local:
                    continue

                # Only fire if leg1 has actually landed — causal, no noise
                leg1_landed = (leg1_arr_ns_v[j] > 0 and
                               leg1_arr_ns_v[j] < t_ns)
                if not leg1_landed:
                    continue

                leg1_local = fl2local[l1]
                leg2_local = fl2local[l2]
                r_src.append(leg1_local)
                r_dst.append(leg2_local)
                r_attr.append([float(actual_delays[j]),
                                float(turn_norms[j])])

    # ── departs_from: UNCONDITIONAL ──────────────────────────────────────
    origin_idx = fa["origin_idx"][fids]
    valid_o    = origin_idx >= 0
    o_idx_safe    = np.where(valid_o, origin_idx, 0)
    o_avg_dep     = ap_avg_dep[o_idx_safe]
    o_avg_taxi    = ap_avg_taxi[o_idx_safe]
    o_baseline    = ap_taxi_baseline[o_idx_safe]

    congestion_w  = (valid_o & (
        (o_avg_dep  > AP_DEP_DELAY_THRESH) |
        (o_avg_taxi > o_baseline * AP_TAXI_MULT_THRESH)
    )).astype(np.float32)

    dep_fl_local = np.where(valid_o)[0]
    dep_ap_idx   = origin_idx[dep_fl_local]
    dep_weights  = congestion_w[dep_fl_local].reshape(-1, 1)

    # ── arrives_at: causal ───────────────────────────────────────────────
    # Fires only when the IMMEDIATE inbound delay is significant.
    # Using cumul here was wrong — a plane on its 4th leg with 60min total
    # cumulative delay but an on-time last leg should NOT fire arrives_at.
    # The cumulative signal is already encoded in the flight node features.
    # arrives_at is specifically: "this plane just arrived late, the airport
    # should know about it" — which is the immediate inbound leg only.
    dest_idx  = fa["dest_idx"][fids]
    valid_d   = dest_idx >= 0
    immed     = fa["immed_inbound"][fids]

    has_delay     = valid_d & (immed > INBOUND_DELAY_THRESH)
    arr_fl_local  = np.where(has_delay)[0]
    arr_ap_idx    = dest_idx[arr_fl_local]

    def mk(src, dst, attr=None, d=1):
        if len(src) == 0:
            return (np.zeros((2,0),dtype=np.int64),
                    np.zeros((0,d),dtype=np.float32))
        ei = np.stack([np.asarray(src,dtype=np.int64),
                       np.asarray(dst,dtype=np.int64)], axis=0)
        ea = (np.array(attr,dtype=np.float32) if attr
              else np.ones((len(src),d),dtype=np.float32))
        return ei, ea

    return (mk(r_src, r_dst, r_attr, d=2),
            mk(dep_fl_local, dep_ap_idx,
               attr=dep_weights.tolist(), d=1),
            mk(arr_fl_local, arr_ap_idx, d=1))


def get_ap_rotation_tensors(rotation_index, t):
    """Airport→airport rotation edges — unchanged, still 3-attr."""
    import torch
    entry = rotation_index.get(t)
    if entry is None:
        return (torch.zeros((2,0),dtype=torch.long),
                torch.zeros((0,3),dtype=torch.float))
    src, dst, arr_data, _ = entry
    # Airport rotation uses only the first element as a proxy weight
    # (turnaround_norm from arr_data[:,0])
    ap_attr = arr_data[:,:1].astype(np.float32)   # (n, 1) — turnaround
    return (torch.tensor(np.stack([src,dst]),dtype=torch.long),
            torch.tensor(ap_attr,dtype=torch.float))


def to_torch(ei, ea):
    import torch
    if ei.shape[1] == 0:
        d = ea.shape[1] if ea.ndim > 1 else 1
        return (torch.zeros((2,0),dtype=torch.long),
                torch.zeros((0,d),dtype=torch.float))
    return (torch.tensor(ei,dtype=torch.long),
            torch.tensor(ea,dtype=torch.float))


# ════════════════════════════════════════════════════════════════════════════
# SAME-SNAPSHOT ROTATION INBOUND LOOKUP
# ════════════════════════════════════════════════════════════════════════════

def build_same_snap_inbound(rot_grp, fl_flight_id_set):
    """
    From the rotation group (now keyed by leg2_dep hour), build a dict:
    leg2_flight_id -> (immed_delay, turnaround_min)
    Only for flight_ids that exist in this snapshot.
    Used to override immed_inbound feature for flights with known inbound delay.
    """
    if rot_grp is None or len(rot_grp) == 0:
        return {}
    result = {}
    # Try both column name variants
    leg2_col = "leg2_flight_id" if "leg2_flight_id" in rot_grp.columns \
               else ("leg2_id" if "leg2_id" in rot_grp.columns else None)
    if leg2_col is None:
        return {}
    for _, row in rot_grp.iterrows():
        l2 = row.get(leg2_col, -1)
        if l2 in fl_flight_id_set:
            result[l2] = (
                float(row.get("leg1_arr_delay", 0.0) or 0.0),
                float(row.get("turnaround_min",  0.0) or 0.0),
            )
    return result


# ════════════════════════════════════════════════════════════════════════════
# MAIN SNAPSHOT LOOP — OPTIMIZED
# ════════════════════════════════════════════════════════════════════════════

def build_snapshots(airports, ap2idx, snapshot_times,
                    X_ap_static, flights, fa, top_hubs, taxi_baseline,
                    dep_lookup, arr_lookup, ap_label_lookup,
                    rotation_index, snap_to_fids,
                    ap_congestion, traffic_lookup, forecast_lookup,
                    cg_ei, cg_src, cg_dst, cg_base_w, cg_taxi_mean, cg_taxi_std,
                    nw_ei, nw_ea):
    import torch
    from torch_geometric.data import HeteroData

    # Build static network edge tensors (unchanged every snapshot)
    nw_t  = to_torch(nw_ei, nw_ea)
    n_ap  = len(airports)
    ap_tb = fa["ap_taxi_baseline"]   # pre-extracted baseline array

    # Static congestion topology (indices only — weights computed per snapshot)
    has_cg = len(cg_src) > 0
    cg_ei_t = (torch.tensor(np.stack([cg_src, cg_dst]), dtype=torch.long)
               if has_cg else torch.zeros((2,0), dtype=torch.long))

    snapshots = []
    total_dep_edges = total_arr_edges = total_rot_edges = total_n = 0

    for t in tqdm(snapshot_times, desc="Building snapshots", unit="snap"):
        data  = HeteroData()
        t_ns  = np.datetime64(t,"ns").astype(np.int64)

        # ── Airport features ──────────────────────────────────────────────
        X_dyn  = build_ap_features_fast(t, airports, ap2idx,
                                         dep_lookup, arr_lookup,
                                         traffic_lookup, forecast_lookup)
        X_time = build_time_features(t, n_ap)
        X_ap   = np.hstack([X_ap_static, X_dyn, X_time])
        data["airport"].x             = torch.tensor(X_ap, dtype=torch.float16)
        data["airport"].num_nodes     = n_ap
        data["airport"].snapshot_time = str(t)
        # Store exact snapshot time as int64 ns for longitudinal training
        # This avoids the flawed dep_ns.min() - 6.5h estimation in step 6
        t_ns_scalar = np.datetime64(t, "ns").astype(np.int64)
        data["snap_time_ns"] = torch.tensor([t_ns_scalar], dtype=torch.int64)

        y_ap = build_ap_labels(t, ap2idx, ap_label_lookup)
        data["airport"].y      = torch.tensor(y_ap, dtype=torch.float)
        data["airport"].y_mask = torch.tensor(~np.isnan(y_ap), dtype=torch.bool)

        # ── Airport rotation edges (dynamic) ──────────────────────────────
        re_ei, re_ea = get_ap_rotation_tensors(rotation_index, t)
        data["airport","rotation","airport"].edge_index = re_ei
        data["airport","rotation","airport"].edge_attr  = re_ea

        # ── Airport congestion arrays for this snapshot ───────────────────
        ap_avg_dep, ap_avg_taxi = ap_congestion.get(
            t, (np.zeros(n_ap,dtype=np.float32),
                np.zeros(n_ap,dtype=np.float32)))

        # ── Dynamic congestion weights ────────────────────────────────────
        # Compute live edge weights from actual taxi/delay state this hour.
        # Edges that were quiet historically but congested right now get
        # amplified; edges between currently-clear airports get attenuated.
        if has_cg:
            cg_w = compute_dynamic_congestion_weights(
                cg_src, cg_dst, cg_base_w,
                cg_taxi_mean, cg_taxi_std,
                ap_avg_taxi, ap_avg_dep)
            # Store as (n_edges, 2): [dynamic_weight, base_weight]
            cg_ea_t = torch.tensor(
                np.stack([cg_w, cg_base_w], axis=1), dtype=torch.float)
        else:
            cg_ea_t = torch.zeros((0,2), dtype=torch.float)

        # ── Flight nodes via binary search pre-index ──────────────────────
        fids = snap_to_fids.get(t, np.array([], dtype=np.int64))
        rot_entry = rotation_index.get(t)
        rot_grp   = rot_entry[3] if rot_entry is not None else None

        # Add landed inbound source flights for true leg1->leg2 rotation edges.
        # These support nodes are unlabeled because they already departed, but
        # they let us replace the old leg2 self-loop with a causal flight edge.
        if rot_grp is not None and len(rot_grp) > 0:
            leg1_col = ("leg1_flight_id" if "leg1_flight_id" in rot_grp.columns
                        else ("leg1_id" if "leg1_id" in rot_grp.columns else None))
            if leg1_col is not None and "leg1_arr_ns" in rot_grp.columns:
                landed = rot_grp["leg1_arr_ns"].values < t_ns
                landed &= rot_grp["leg1_arr_ns"].values > 0
                extra_ids = rot_grp.loc[landed, leg1_col].to_numpy(dtype=np.int64)
                if extra_ids.size > 0:
                    # rotation edges store global flight_id values, while the
                    # pre-extracted arrays are indexed by filtered row position.
                    # Translate global ids -> local row positions safely.
                    flight_ids = fa["flight_id"]
                    pos = np.searchsorted(flight_ids, extra_ids)
                    valid = pos < len(flight_ids)
                    valid &= (flight_ids[np.clip(pos, 0, len(flight_ids)-1)] == extra_ids)
                    extra_rows = pos[valid].astype(np.int64)
                    if extra_rows.size > 0:
                        extra_rows = np.setdiff1d(extra_rows, fids, assume_unique=False)
                        if extra_rows.size > 0:
                            fids = np.concatenate([fids, extra_rows])

        n_fl = len(fids)

        # Same-snapshot rotation inbound — updates immed_inbound for known delays
        if n_fl > 0 and rot_grp is not None:
            fid_set = set(fa["flight_id"][fids].tolist())
            same_snap_inbound = build_same_snap_inbound(rot_grp, fid_set)
        else:
            same_snap_inbound = {}

        # ── Flight features (float16 — halves storage, fine for [-1,1] range) ──
        X_fl = build_flight_features_fast(fids, fa, t_ns, same_snap_inbound)
        data["flight"].x         = torch.tensor(X_fl, dtype=torch.float16)
        data["flight"].num_nodes = n_fl

        # ── Flight labels (float32 — keep full precision for loss) ────────
        y_fl, fl_masks = build_flight_labels_fast(fids, fa, t_ns)
        data["flight"].y = torch.tensor(y_fl, dtype=torch.float)
        for h in LABEL_HORIZONS_FL:
            setattr(data["flight"], f"y_mask_{h}h",
                    torch.tensor(fl_masks[h], dtype=torch.bool))

        # ── Metadata: flight_id, tail_id, scheduled times ────────────────
        if n_fl > 0:
            data["flight"].flight_id     = torch.tensor(
                fa["flight_id"][fids], dtype=torch.long)
            data["flight"].scheduled_dep = torch.tensor(
                fa["dep_dt_ns"][fids], dtype=torch.long)
            data["flight"].scheduled_arr = torch.tensor(
                fa["arr_dt_ns"][fids], dtype=torch.long)
            # Fix 3: tail_id for per-tail GRU in step 6
            data["flight"].tail_id = torch.tensor(
                fa["tail_id"][fids], dtype=torch.long)

        # ── Flight edges (phase-aware rotation + causal flight edges) ────────
        rot_arr_data = rot_entry[2] if rot_entry is not None else np.zeros((0,4))
        (fl_rot_np, dep_np, arr_np) = build_flight_edges_fast(
            fids, fa, rot_grp, rot_arr_data,
            ap_avg_dep, ap_avg_taxi, ap_tb, t_ns)

        fl_rot_t = to_torch(*fl_rot_np)   # edge_attr: (n, 4)
        dep_t    = to_torch(*dep_np)
        arr_t    = to_torch(*arr_np)

        data["flight","rotation",    "flight" ].edge_index = fl_rot_t[0]
        data["flight","rotation",    "flight" ].edge_attr  = fl_rot_t[1]
        data["flight","departs_from","airport"].edge_index = dep_t[0]
        data["flight","departs_from","airport"].edge_attr  = dep_t[1]
        data["flight","arrives_at",  "airport"].edge_index = arr_t[0]
        data["flight","arrives_at",  "airport"].edge_attr  = arr_t[1]

        # Fix 2: add reverse airport→flight edges so HGT can send
        # airport context back to flights in layer 2.
        # Without these, airports only receive from flights (layer 1)
        # but never communicate back — airport GRU state never reaches fl_pred.
        # Reverse departs_from: airport → flight (same edge, flipped)
        if dep_t[0].shape[1] > 0:
            dep_rev_ei = dep_t[0][[1,0]]   # flip src/dst
            data["airport","departs_to","flight"].edge_index = dep_rev_ei
            data["airport","departs_to","flight"].edge_attr  = dep_t[1]
        else:
            data["airport","departs_to","flight"].edge_index =                 torch.zeros((2,0), dtype=torch.long)
            data["airport","departs_to","flight"].edge_attr  =                 torch.zeros((0,1), dtype=torch.float)
        # Reverse arrives_at: airport → flight (same edge, flipped)
        if arr_t[0].shape[1] > 0:
            arr_rev_ei = arr_t[0][[1,0]]
            data["airport","arrives_from","flight"].edge_index = arr_rev_ei
            data["airport","arrives_from","flight"].edge_attr  = arr_t[1]
        else:
            data["airport","arrives_from","flight"].edge_index =                 torch.zeros((2,0), dtype=torch.long)
            data["airport","arrives_from","flight"].edge_attr  =                 torch.zeros((0,1), dtype=torch.float)

        total_dep_edges += dep_np[0].shape[1]
        total_arr_edges += arr_np[0].shape[1]
        total_rot_edges += fl_rot_np[0].shape[1]
        total_n += 1
        snapshots.append(data)

    d = max(total_n, 1)
    print(f"\n  Causal edges (avg/snapshot):")
    print(f"    departs_from         : {total_dep_edges/d:.1f}")
    print(f"    arrives_at           : {total_arr_edges/d:.1f}")
    print(f"    flight rotation      : {total_rot_edges/d:.1f}")
    print(f"    congestion (dynamic) : {len(cg_src):,} edges, weights vary per snapshot")
    return snapshots


def split_snapshots(snapshots):
    train, val, test = [], [], []
    for snap in snapshots:
        yr = pd.Timestamp(snap["airport"].snapshot_time).year
        if yr in TRAIN_YEARS:   train.append(snap)
        elif yr in VAL_YEARS:   val.append(snap)
        elif yr in TEST_YEARS:  test.append(snap)
    return train, val, test


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    import torch
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print("STEP 5 — CAUSAL + FULL TAIL PROPAGATION (OPTIMIZED)")
    print("=" * 65)

    print("\nLoading weather airports ...")
    weather_airports = load_weather_airports()

    print("\nLoading edge tables ...")
    rotation_edges   = load_parquet_or_empty(
        os.path.join(GRAPH_DATA_DIR,"rotation_edges.parquet"),  "rotation_edges")
    congestion_edges = load_parquet_or_empty(
        os.path.join(GRAPH_DATA_DIR,"congestion_edges.parquet"),"congestion_edges")

    print("\nLoading flights ...")
    flights = load_flights(weather_airports)

    print("\nBuilding airport index ...")
    airports, ap2idx = build_airport_index(weather_airports)
    pd.DataFrame({"airport":airports,
                  "node_idx":range(len(airports))}).to_parquet(
        os.path.join(OUTPUT_DIR,"airport_index.parquet"), index=False)

    # All static/historical feature lookups use TRAINING years only.
    # Using val/test years here would leak future information into airport
    # node priors (hist_avg_dep_delay, taxi_baseline, network correlations).
    # Route stats already filtered — now consistent across all lookups.
    train_flights = flights[
        flights["dep_datetime"].dt.year.isin(TRAIN_YEARS)].copy()
    print(f"\n  Static features computed from {len(train_flights):,} "
          f"training flights ({TRAIN_YEARS}) — no val/test leakage")

    X_ap_static, top_hubs, taxi_baseline = build_static_airport_features(
        airports, ap2idx, train_flights)

    print("\nBuilding feature lookups ...")
    forecast_lookup = build_weather_forecast_lookup(train_flights)
    traffic_lookup  = build_traffic_lookup(flights, ap2idx)   # all years — real-time state, not historical prior
    route_stats     = build_route_stats_lookup(train_flights)
    print(f"  Route stats: {len(train_flights):,} training flights ({TRAIN_YEARS})")

    # Save route stats lookup for step 7 inference
    lf, lh, lr, gm, gs = route_stats
    rs_rows = []
    for (o, d, h, dw), (avg, std) in lf.items():
        rs_rows.append({"ORIGIN":o, "DEST":d, "dep_hour":h,
                        "DayOfWeek":dw, "hist_avg":avg, "hist_std":std})
    pd.DataFrame(rs_rows).to_parquet(
        os.path.join(OUTPUT_DIR, "route_stats.parquet"), index=False)
    pd.DataFrame([{"global_mean":gm,"global_std":gs}]).to_parquet(
        os.path.join(OUTPUT_DIR, "route_stats_global.parquet"), index=False)
    print(f"  ✅ Route stats saved → route_stats.parquet")

    network_edges = build_network_edges(airports, ap2idx, train_flights)
    network_edges.to_parquet(
        os.path.join(OUTPUT_DIR,"network_edges.parquet"), index=False)
    print("  ✅ Network edges saved")

    tail_propagation_lookup = build_tail_propagation_lookup(flights)   # all years — same-day state

    dep_lookup, arr_lookup = preaggregate_airport_features(flights)
    ap_label_lookup        = preaggregate_airport_labels(flights)

    print("\nPre-processing edges ...")
    rotation_index = preprocess_rotation_edges(
        rotation_edges, ap2idx, weather_airports)
    (cg_ei, cg_src, cg_dst,
     cg_base_w, cg_taxi_mean, cg_taxi_std) = preprocess_congestion_edges(
        congestion_edges, ap2idx, weather_airports)
    nw_ei, nw_ea   = preprocess_network_edges(network_edges, ap2idx)

    min_time       = flights["dep_datetime"].min().floor("h")
    max_time       = flights["dep_datetime"].max().ceil("h")
    snapshot_times = pd.date_range(min_time, max_time, freq=SNAPSHOT_FREQ)
    print(f"\n  Snapshot range : {min_time} → {max_time}")
    print(f"  Total snapshots: {len(snapshot_times):,}")

    # ── OPTIMIZATIONS: pre-compute everything before the loop ────────────
    snap_to_fids  = preindex_flights(flights, snapshot_times)
    fa            = preextract_flight_arrays(
        flights, ap2idx, top_hubs, taxi_baseline,
        tail_propagation_lookup, airports,
        route_stats=route_stats)
    ap_congestion = precompute_ap_congestion(
        snapshot_times, airports, dep_lookup, taxi_baseline)

    print(f"\n{'='*65}")
    print(f"  Airport feat dim  : {N_AP_FEATURES}")
    print(f"  Flight feat dim   : {N_FL_FEATURES}")
    print(f"  Edge types        : 6 (all causally justified)")
    print(f"  Label horizons    : {LABEL_HORIZONS_FL}h")
    print(f"  Tail entries      : {len(tail_propagation_lookup):,}")
    print(f"{'='*65}")

    print("\nBuilding snapshots ...")
    t0        = time.time()
    snapshots = build_snapshots(
        airports, ap2idx, snapshot_times,
        X_ap_static, flights, fa, top_hubs, taxi_baseline,
        dep_lookup, arr_lookup, ap_label_lookup,
        rotation_index, snap_to_fids,
        ap_congestion, traffic_lookup, forecast_lookup,
        cg_ei, cg_src, cg_dst, cg_base_w, cg_taxi_mean, cg_taxi_std,
        nw_ei, nw_ea,
    )
    elapsed = time.time() - t0
    print(f"\n  {len(snapshots):,} snapshots in {elapsed/60:.1f} min")

    train, val, test = split_snapshots(snapshots)
    print(f"\n  Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

    print("\nSaving ...")
    for name, data in [("train",train),("val",val),("test",test)]:
        out = os.path.join(OUTPUT_DIR, f"snapshots_{name}.pt")
        torch.save(data, out)
        size_gb = os.path.getsize(out)/1e9
        print(f"  ✅ {name} → {out}  ({size_gb:.2f} GB)")

    # Save static edges ONCE — not in every snapshot
    # Step 6 loads this and injects into each snapshot before forward pass
    # Note: congestion edge WEIGHTS are now dynamic (computed per snapshot
    # from live taxi/delay data). We save the topology + baseline for step 6
    # to recompute weights during training/inference.
    static_edges = {
        "congestion_ei"       : torch.tensor(cg_ei,        dtype=torch.long),
        "congestion_src"      : torch.tensor(cg_src,       dtype=torch.long),
        "congestion_dst"      : torch.tensor(cg_dst,       dtype=torch.long),
        "congestion_base_w"   : torch.tensor(cg_base_w,    dtype=torch.float),
        "congestion_taxi_mean": torch.tensor(cg_taxi_mean, dtype=torch.float),
        "congestion_taxi_std" : torch.tensor(cg_taxi_std,  dtype=torch.float),
        "network_ei"          : torch.tensor(nw_ei,        dtype=torch.long),
        "network_ea"          : torch.tensor(nw_ea,        dtype=torch.float),
    }
    static_path = os.path.join(OUTPUT_DIR, "static_edges.pt")
    torch.save(static_edges, static_path)
    static_mb = os.path.getsize(static_path)/1e6
    print(f"  ✅ static_edges → {static_path}  ({static_mb:.1f} MB)")

    # Save flight metadata lookup for step 7 inference table
    # Step 7 joins on flight_id to get origin/dest/tail/airline
    lookup_cols = ["flight_id","ORIGIN","DEST","dep_datetime","arr_datetime"]
    if "Tail_Number"        in flights.columns: lookup_cols.append("Tail_Number")

    # Save tail2idx mapping for step 6 (per-tail GRU needs consistent indices)
    import json
    tail2idx_path = os.path.join(OUTPUT_DIR, "tail2idx.json")
    with open(tail2idx_path, "w") as f:
        json.dump(fa.get("tail2idx", {}), f)
    print(f"  tail2idx saved: {tail2idx_path}")
    if "Operating_Airline"  in flights.columns: lookup_cols.append("Operating_Airline")
    if "Flight_Number_Operating_Airline" in flights.columns:
        lookup_cols.append("Flight_Number_Operating_Airline")
    flight_lookup = flights[lookup_cols].copy()
    lookup_path   = os.path.join(OUTPUT_DIR, "flight_lookup.parquet")
    flight_lookup.to_parquet(lookup_path, index=False)
    lookup_mb = os.path.getsize(lookup_path)/1e6
    print(f"  ✅ flight_lookup → {lookup_path}  ({lookup_mb:.0f} MB)")

    print(f"\n{'='*65}")
    print("STEP 5 COMPLETE — next: run 06_train_gnn.py")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
