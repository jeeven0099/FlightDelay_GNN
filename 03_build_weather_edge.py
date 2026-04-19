"""
STEP 3 — BUILD WEATHER EDGES
Strategy:
  1. Load all weather files, attach airport code from filename.
  2. For each flight departure, join the nearest-hour weather record
     at the origin airport → node feature (not an edge by itself).
  3. Build WEATHER EDGES between airports that share a weather system:
       - Spatial proximity (haversine < SPATIAL_KM_THRESHOLD km), OR
       - Correlated severe-weather windows (both airports report
         visibility < VIS_THRESHOLD or precip > PRECIP_THRESHOLD
         within the same hour)
  4. Edge features: wind_speed, visibility, precip, pressure_delta,
     temporal_overlap_score.

Output:
  weather_node_features.parquet   — per-flight weather snapshot at origin
  weather_edges.parquet           — airport-airport weather edges

Usage:
    python 03_build_weather_edges.py
"""

import os, glob, re
import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt

# ── CONFIG ──────────────────────────────────────────────────────────────────
FLIGHTS_PATH      = r"C:\Users\user\Desktop\Airline_Graphs_Project"
WEATHER_DIR       = r"C:\Users\user\Desktop\Airline_Graphs_Project\data\datasets\noaa_weather_data\noaa_weather_parsed"
OUTPUT_DIR        = r"C:\Users\user\Desktop\Airline_Graphs_Project\graph_data"

SPATIAL_KM_THRESHOLD  = 35    # airports within this distance share weather edges
VIS_THRESHOLD_M       = 4800   # < 3 statute miles → low visibility event
PRECIP_THRESHOLD_MM   = 2.0    # ≥ 2 mm precip/hour → significant precip
WIND_THRESHOLD_MS     = 10.0   # ≥ 10 m/s → high wind event
WEATHER_JOIN_TOL_H    = 1      # hours: snap flight dep_time to nearest weather obs
# ────────────────────────────────────────────────────────────────────────────

WEATHER_COLS = [
    "datetime", "latitude", "longitude", "elevation_m",
    "temp_c", "dewpoint_c", "relative_humidity_pct",
    "wind_speed_ms", "wind_gust_ms", "wind_dir_deg",
    "sea_level_pressure_hpa", "visibility_m",
    "ceiling_m", "sky_cover_oktas",
    "precip_depth_mm", "snow_depth_mm",
    "station_id", "wban",
]


# ── Haversine distance ───────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))


def find_flight_file(root):
    for ext in ("*.csv", "*.parquet", "*.feather"):
        m = glob.glob(os.path.join(root, ext))
        if m:
            return m[0]
    raise FileNotFoundError(f"No flight file in {root}")


def find_weather_files(weather_dir):
    """Return dict: (AIRPORT, YEAR) → filepath"""
    pattern = re.compile(r"([A-Z]{3,4})_(\d{4})\.(csv|csc)$", re.IGNORECASE)
    index = {}
    for root, _, files in os.walk(weather_dir):
        for f in files:
            m = pattern.match(f)
            if m:
                ap   = m.group(1).upper()
                year = int(m.group(2))
                index[(ap, year)] = os.path.join(root, f)
    return index


def load_weather_file(fp):
    """Load one weather CSV, keep only useful columns."""
    try:
        df = pd.read_csv(fp, low_memory=False)
    except Exception as e:
        print(f"    ⚠ Could not read {fp}: {e}")
        return None

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Map to standard names (files may have slight column name variations)
    rename_map = {
        "wind_speed_ms":        ["wind_speed_ms", "wind_speed", "wspd"],
        "wind_gust_ms":         ["wind_gust_ms",  "wind_gust",  "wgst"],
        "visibility_m":         ["visibility_m",  "visibility", "vis"],
        "precip_depth_mm":      ["precip_depth_mm","precip_depth","prcp"],
        "sea_level_pressure_hpa": ["sea_level_pressure_hpa","slp"],
        "ceiling_m":            ["ceiling_m", "ceiling", "ceil"],
    }
    actual = {}
    for standard, candidates in rename_map.items():
        for c in candidates:
            if c in df.columns:
                actual[c] = standard
                break
    df = df.rename(columns={v: k for k, v in actual.items()})

    # Parse datetime
    dt_col = next((c for c in df.columns if "datetime" in c), None)
    if dt_col:
        df["datetime"] = pd.to_datetime(df[dt_col], dayfirst=True, errors="coerce")
    else:
        return None

    keep = [c for c in WEATHER_COLS if c in df.columns]
    return df[keep].dropna(subset=["datetime"])


def load_all_weather(weather_index):
    """Load all weather files, attach airport code, concatenate."""
    print(f"\nLoading {len(weather_index)} weather files ...")
    frames = []
    for (ap, year), fp in weather_index.items():
        df = load_weather_file(fp)
        if df is not None and len(df) > 0:
            df["airport"] = ap
            df["file_year"] = year
            frames.append(df)
        else:
            print(f"    ⚠ Empty or unreadable: {os.path.basename(fp)}")

    if not frames:
        raise RuntimeError("No weather data loaded. Check WEATHER_DIR path.")

    weather = pd.concat(frames, ignore_index=True)
    weather = weather.sort_values(["airport", "datetime"])
    print(f"  Loaded {len(weather):,} weather observations across "
          f"{weather['airport'].nunique()} airports.")
    return weather


def build_airport_locations(weather):
    """Derive one (lat, lon) per airport from median of observations."""
    loc_cols = [c for c in ["latitude", "longitude", "elevation_m"] if c in weather.columns]
    locs = (weather.groupby("airport")[loc_cols]
                   .median()
                   .reset_index()
                   .rename(columns={"latitude": "lat", "longitude": "lon"}))
    print(f"\n  Airport locations derived for {len(locs)} airports.")
    return locs


def join_weather_to_flights(flights, weather):
    """
    For each flight, find the closest weather observation at ORIGIN
    within WEATHER_JOIN_TOL_H hours before departure.
    Returns DataFrame with weather features appended to flight rows.
    """
    print("\nJoining weather to flights ...")

    # Round flight dep_datetime to nearest hour for fast merge
    flights = flights.copy()
    flights["dep_hour"] = flights["dep_datetime"].dt.floor("h")

    weather_hr = weather.copy()
    weather_hr["obs_hour"] = weather_hr["datetime"].dt.floor("h")

    # Build lookup: (airport, hour) → weather record
    weather_agg = (weather_hr
                   .groupby(["airport", "obs_hour"])
                   .agg({
                       "wind_speed_ms":          "mean",
                       "wind_gust_ms":            "max",
                       "visibility_m":            "min",
                       "precip_depth_mm":         "sum",
                       "sea_level_pressure_hpa":  "mean",
                       "ceiling_m":               "min",
                       "sky_cover_oktas":         "max",
                       "temp_c":                  "mean",
                   })
                   .reset_index()
                   .rename(columns={"airport": "ORIGIN",
                                    "obs_hour": "dep_hour"}))

    # Left-merge on ORIGIN + dep_hour
    merged = flights.merge(weather_agg, on=["ORIGIN", "dep_hour"], how="left")

    coverage = merged["wind_speed_ms"].notna().mean()
    print(f"  Weather join coverage: {coverage*100:.1f}% of flights have weather data")

    return merged


def flag_severe_weather(weather):
    """Add boolean columns for adverse weather events."""
    w = weather.copy()
    w["low_vis"]     = w.get("visibility_m",    pd.Series(dtype=float)).lt(VIS_THRESHOLD_M)
    w["high_wind"]   = w.get("wind_speed_ms",   pd.Series(dtype=float)).ge(WIND_THRESHOLD_MS)
    w["heavy_precip"]= w.get("precip_depth_mm", pd.Series(dtype=float)).ge(PRECIP_THRESHOLD_MM)
    w["any_severe"]  = w["low_vis"] | w["high_wind"] | w["heavy_precip"]
    return w


def build_weather_edges(weather, airport_locs):
    """
    Build weather edges between airports in the same metro area
    (spatial proximity only, co-severe edges removed).
    Edge weight = 1.0 for all pairs — same-metro airports share
    identical weather by definition, no gradient needed.
    """
    print("\nBuilding weather edges ...")
    airports = airport_locs["airport"].tolist()
    n = len(airports)

    print(f"  Computing pairwise distances for {n} airports ...")
    loc_dict = airport_locs.set_index("airport")[["lat", "lon"]].to_dict("index")

    spatial_edges = []
    for i, ap1 in enumerate(airports):
        for j in range(i + 1, n):
            ap2 = airports[j]
            if ap1 not in loc_dict or ap2 not in loc_dict:
                continue
            d = haversine_km(
                loc_dict[ap1]["lat"], loc_dict[ap1]["lon"],
                loc_dict[ap2]["lat"], loc_dict[ap2]["lon"],
            )
            if d <= SPATIAL_KM_THRESHOLD:
                spatial_edges.append({
                    "edge_type"   : "weather",
                    "src_airport" : ap1,
                    "dst_airport" : ap2,
                    "distance_km" : round(d, 1),
                    "edge_weight" : 1.0,  # same metro = full weight
                })

    print(f"  Spatial proximity edges (<{SPATIAL_KM_THRESHOLD} km): "
          f"{len(spatial_edges):,}")

    if not spatial_edges:
        print("  ⚠ No weather edges found.")
        return pd.DataFrame()

    edges = pd.DataFrame(spatial_edges)

    print(f"\n  Final weather edges: {len(edges)}")
    for _, row in edges.iterrows():
        print(f"    {row['src_airport']} — {row['dst_airport']}: "
              f"{row['distance_km']} km")

    return edges


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load flights ───────────────────────────────────────────────────────
    fp = find_flight_file(FLIGHTS_PATH)
    ext = os.path.splitext(fp)[1].lower()
    if ext == ".csv":
        flights = pd.read_csv(fp, low_memory=False)
    else:
        flights = pd.read_parquet(fp)
    for col in ("dep_datetime", "arr_datetime"):
        if col in flights.columns:
            flights[col] = pd.to_datetime(flights[col], dayfirst=True, errors="coerce")
    print(f"Flights loaded: {len(flights):,} rows")

    # ── Load weather ───────────────────────────────────────────────────────
    weather_index = find_weather_files(WEATHER_DIR)
    print(f"Weather files found: {len(weather_index)}")
    weather = load_all_weather(weather_index)

    # ── Airport locations ──────────────────────────────────────────────────
    airport_locs = build_airport_locations(weather)
    # ── Manual coordinate corrections ──────────────────────────────────────────
    # These airports had mismatched weather station files giving wrong lat/lon
    COORD_OVERRIDES = {
        "FLL": (26.073, -80.150),
        "SLC": (40.788, -111.978),
        "CLE": (41.411, -81.849),
        "TPA": (27.975, -82.533),
        "CMH": (39.998, -82.892),
    }
    for ap, (lat, lon) in COORD_OVERRIDES.items():
        mask = airport_locs["airport"] == ap
        airport_locs.loc[mask, "lat"] = lat
        airport_locs.loc[mask, "lon"] = lon
        print(f"  Corrected coordinates for {ap}: ({lat}, {lon})")
# ────────────────────────────────────────────────────────────────────────────
    # ── Join weather to flights (node feature table) ───────────────────────
    flights_with_wx = join_weather_to_flights(flights, weather)
    out1 = os.path.join(OUTPUT_DIR, "weather_node_features.parquet")
    flights_with_wx.to_parquet(out1, index=False)
    print(f"\n✅  Saved node features → {out1}")

    # ── Build weather edges ────────────────────────────────────────────────
    weather_edges = build_weather_edges(weather, airport_locs)
    if len(weather_edges) > 0:
        out2 = os.path.join(OUTPUT_DIR, "weather_edges.parquet")
        weather_edges.to_parquet(out2, index=False)
        print(f"✅  Saved weather edges → {out2}")

    out3 = os.path.join(OUTPUT_DIR, "airport_locations.parquet")
    airport_locs.to_parquet(out3, index=False)
    print(f"✅  Saved airport locations → {out3}")

    print("\nNext: run  04_build_congestion_edges.py")


if __name__ == "__main__":
    main()