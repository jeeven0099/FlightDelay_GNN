"""
STEP 8 — REAL-TIME API CONNECTOR
==================================
Fetches live flight + weather data every 30 minutes and runs the
GNN model to produce fresh predictions.

FREE APIs USED:
  1. OpenSky Network  — live flight positions + tail numbers (ICAO24)
     https://opensky-network.org/api  (no key needed for basic access)

  2. NWS Weather API  — current conditions + 6h forecast at all 36 airports
     https://api.weather.gov  (no key needed, US only)

  3. AviationStack    — flight schedules + delay status + tail numbers
     https://aviationstack.com  (free tier: 100 calls/month)
     Set AVIATIONSTACK_KEY in environment or .env file

USAGE:
  # Run once to get current predictions
  python 08_realtime_connector.py --mode predict

  # Launch live dashboard (refreshes every 30 min)
  python 08_realtime_connector.py --mode dashboard

  # Test API connections
  python 08_realtime_connector.py --mode test

REQUIRES:
  pip install requests python-dotenv dash plotly torch torch-geometric
"""

import os
import time
import argparse
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

# ── CONFIG ──────────────────────────────────────────────────────────────────
BASE_DIR       = r"C:\Users\user\Desktop\Airline_Graphs_Project"
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
GRAPH_DATA_DIR = os.path.join(BASE_DIR, "graph_data")

# API keys (set in .env file or environment variables)
AVIATIONSTACK_KEY = os.getenv("AVIATIONSTACK_KEY", "")

# Refresh interval
REFRESH_MINUTES = 30

# The 36 hub airports — IATA codes and their ICAO equivalents
AIRPORTS = {
    "ANC":"PANC","ATL":"KATL","BNA":"KBNA","BOS":"KBOS",
    "BWI":"KBWI","CLE":"KCLE","CLT":"KCLT","CMH":"KCMH",
    "DEN":"KDEN","DFW":"KDFW","DTW":"KDTW","EWR":"KEWR",
    "FLL":"KFLL","HOU":"KHOU","IAD":"KIAD","IAH":"KIAH",
    "IND":"KIND","JFK":"KJFK","LAS":"KLAS","LAX":"KLAX",
    "LGA":"KLGA","MCI":"KMCI","MCO":"KMCO","MIA":"KMIA",
    "MKE":"KMKE","MSP":"KMSP","ORD":"KORD","PHL":"KPHL",
    "PHX":"KPHX","PIT":"KPIT","SAN":"KSAN","SEA":"KSEA",
    "SFO":"KSFO","SJC":"KSJC","SLC":"KSLC","TPA":"KTPA",
}

AIRPORT_COORDS = {
    "ANC":(61.174,-149.996),"ATL":(33.640,-84.427),
    "BNA":(36.124,-86.678), "BOS":(42.365,-71.009),
    "BWI":(39.175,-76.668), "CLE":(41.411,-81.849),
    "CLT":(35.214,-80.943), "CMH":(39.998,-82.892),
    "DEN":(39.856,-104.674),"DFW":(32.897,-97.038),
    "DTW":(42.212,-83.353), "EWR":(40.692,-74.174),
    "FLL":(26.072,-80.150), "HOU":(29.645,-95.279),
    "IAD":(38.944,-77.456), "IAH":(29.984,-95.341),
    "IND":(39.717,-86.294), "JFK":(40.639,-73.779),
    "LAS":(36.080,-115.152),"LAX":(33.943,-118.408),
    "LGA":(40.777,-73.873), "MCI":(39.298,-94.714),
    "MCO":(28.429,-81.309), "MIA":(25.796,-80.287),
    "MKE":(42.947,-87.897), "MSP":(44.882,-93.222),
    "ORD":(41.978,-87.905), "PHL":(39.872,-75.241),
    "PHX":(33.437,-112.008),"PIT":(40.492,-80.233),
    "SAN":(32.734,-117.190),"SEA":(47.449,-122.309),
    "SFO":(37.619,-122.375),"SJC":(37.363,-121.929),
    "SLC":(40.788,-111.978),"TPA":(27.975,-82.533),
}
# ────────────────────────────────────────────────────────────────────────────


# ════════════════════════════════════════════════════════════════════════════
# 1. OPENSKY NETWORK — free, no key, real-time positions + ICAO24
# ════════════════════════════════════════════════════════════════════════════

def fetch_opensky_flights(airport_iata):
    """
    Fetch current departures from an airport via OpenSky Network.
    Returns DataFrame with: callsign, icao24, dep_airport, arr_airport,
                            dep_time, arr_time, on_ground, velocity
    """
    icao = AIRPORTS.get(airport_iata, "")
    if not icao:
        return pd.DataFrame()

    url = "https://opensky-network.org/api/flights/departure"
    now = int(time.time())
    params = {
        "airport": icao,
        "begin":   now - 7200,   # last 2 hours
        "end":     now,
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if not data:
                return pd.DataFrame()
            df = pd.DataFrame(data)
            df["dep_airport_iata"] = airport_iata
            return df
        elif r.status_code == 429:
            print(f"  ⚠ OpenSky rate limit — waiting 60s")
            time.sleep(60)
            return pd.DataFrame()
        else:
            return pd.DataFrame()
    except Exception as e:
        print(f"  ⚠ OpenSky error for {airport_iata}: {e}")
        return pd.DataFrame()


def fetch_opensky_states():
    """
    Fetch all current airspace states over the continental US.
    Returns DataFrame with: icao24, callsign, origin_country,
                            longitude, latitude, altitude, velocity
    """
    url = "https://opensky-network.org/api/states/all"
    params = {
        "lamin": 24.0, "lamax": 50.0,   # continental US bounding box
        "lomin":-125.0,"lomax":-65.0,
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code == 200:
            data = r.json()
            states = data.get("states", [])
            if not states:
                return pd.DataFrame()
            cols = ["icao24","callsign","origin_country","time_position",
                    "last_contact","longitude","latitude","geo_altitude",
                    "on_ground","velocity","true_track","vertical_rate",
                    "sensors","baro_altitude","squawk","spi","position_source"]
            df = pd.DataFrame(states, columns=cols[:len(states[0])])
            df["icao24"]   = df["icao24"].str.strip()
            df["callsign"] = df["callsign"].str.strip()
            print(f"  OpenSky: {len(df):,} aircraft currently tracked over US")
            return df
        else:
            print(f"  ⚠ OpenSky states error: {r.status_code}")
            return pd.DataFrame()
    except Exception as e:
        print(f"  ⚠ OpenSky states error: {e}")
        return pd.DataFrame()


def icao24_to_tail(icao24):
    """
    Convert ICAO24 hex code to N-number (US tail number).
    Uses OpenSky aircraft database for lookup.
    """
    url = f"https://opensky-network.org/api/metadata/aircraft/icao/{icao24}"
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            return data.get("registration", "")
    except Exception:
        pass
    return ""


def fetch_opensky_all_airports():
    """
    Fetch current departures for all 36 hub airports.
    Rate limited — adds delay between calls.
    """
    print("Fetching OpenSky departures for all 36 airports ...")
    all_flights = []
    for i, (iata, icao) in enumerate(AIRPORTS.items()):
        flights = fetch_opensky_flights(iata)
        if len(flights) > 0:
            all_flights.append(flights)
        if i % 5 == 4:
            time.sleep(2)   # rate limit: max 10 req/10s for anonymous

    if not all_flights:
        return pd.DataFrame()

    df = pd.concat(all_flights, ignore_index=True)
    print(f"  OpenSky: {len(df):,} recent departures across 36 airports")
    return df


# ════════════════════════════════════════════════════════════════════════════
# 2. NWS WEATHER API — free, no key, US airports only
# ════════════════════════════════════════════════════════════════════════════

# NWS station IDs for each airport (K + IATA in most cases)
NWS_STATIONS = {ap: f"K{ap}" for ap in AIRPORTS}

def fetch_nws_current(airport_iata):
    """
    Fetch current weather observation for an airport.
    Returns dict with: temperature, wind_speed, visibility, precip
    """
    station = NWS_STATIONS.get(airport_iata, f"K{airport_iata}")
    url = f"https://api.weather.gov/stations/{station}/observations/latest"
    headers = {"User-Agent": "FlightDelayGNN/1.0 (research)"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            props = r.json().get("properties", {})
            return {
                "airport"     : airport_iata,
                "wind_speed_ms": _safe_val(props.get("windSpeed",{})
                                           .get("value"), 0.0),
                "visibility_m" : _safe_val(props.get("visibility",{})
                                           .get("value"), 10000.0),
                "precip_mm"    : _safe_val(props.get("precipitationLastHour",{})
                                           .get("value"), 0.0),
                "temperature_c": _safe_val(props.get("temperature",{})
                                           .get("value"), 20.0),
                "timestamp"    : props.get("timestamp",""),
            }
        return _default_weather(airport_iata)
    except Exception as e:
        print(f"  ⚠ NWS error for {airport_iata}: {e}")
        return _default_weather(airport_iata)


def fetch_nws_forecast(airport_iata):
    """
    Fetch 6-hour ahead weather forecast for an airport.
    Returns dict with: wind_3h, precip_3h, vis_3h, wind_6h, precip_6h, vis_6h
    """
    lat, lon = AIRPORT_COORDS.get(airport_iata, (35.0, -90.0))
    headers  = {"User-Agent": "FlightDelayGNN/1.0 (research)"}

    # Step 1: get grid point
    try:
        r = requests.get(
            f"https://api.weather.gov/points/{lat},{lon}",
            headers=headers, timeout=10)
        if r.status_code != 200:
            return _default_forecast(airport_iata)
        props = r.json().get("properties", {})
        grid_url = props.get("forecastHourly","")

        if not grid_url:
            return _default_forecast(airport_iata)

        # Step 2: get hourly forecast
        r2 = requests.get(grid_url, headers=headers, timeout=10)
        if r2.status_code != 200:
            return _default_forecast(airport_iata)

        periods = r2.json().get("properties",{}).get("periods",[])
        if len(periods) < 7:
            return _default_forecast(airport_iata)

        def wind_ms(period):
            # NWS gives wind as "15 mph" string
            try:
                speed_str = period.get("windSpeed","0 mph").split()[0]
                return float(speed_str) * 0.44704   # mph → m/s
            except Exception:
                return 0.0

        return {
            "airport"  : airport_iata,
            "wind_3h"  : wind_ms(periods[3]),
            "precip_3h": 1.0 if periods[3].get("probabilityOfPrecipitation",
                          {}).get("value",0) > 40 else 0.0,
            "vis_3h"   : 10000.0,   # NWS hourly doesn't give visibility
            "wind_6h"  : wind_ms(periods[6]),
            "precip_6h": 1.0 if periods[6].get("probabilityOfPrecipitation",
                          {}).get("value",0) > 40 else 0.0,
            "vis_6h"   : 10000.0,
        }
    except Exception as e:
        print(f"  ⚠ NWS forecast error for {airport_iata}: {e}")
        return _default_forecast(airport_iata)


def fetch_all_weather():
    """Fetch current + forecast weather for all 36 airports."""
    print("Fetching NWS weather for all 36 airports ...")
    current  = []
    forecast = []
    for i, iata in enumerate(AIRPORTS):
        current.append(fetch_nws_current(iata))
        forecast.append(fetch_nws_forecast(iata))
        if i % 6 == 5:
            time.sleep(1)   # be polite to NWS API
    curr_df = pd.DataFrame(current)
    fore_df = pd.DataFrame(forecast)
    print(f"  Weather fetched for {len(curr_df)} airports")
    return curr_df, fore_df


def _safe_val(v, default):
    return float(v) if v is not None else default

def _default_weather(airport):
    return {"airport":airport,"wind_speed_ms":0.0,"visibility_m":10000.0,
            "precip_mm":0.0,"temperature_c":20.0,"timestamp":""}

def _default_forecast(airport):
    return {"airport":airport,"wind_3h":0.0,"precip_3h":0.0,"vis_3h":10000.0,
            "wind_6h":0.0,"precip_6h":0.0,"vis_6h":10000.0}


# ════════════════════════════════════════════════════════════════════════════
# 3. AVIATIONSTACK — flight schedules + delay status (free tier: 100 calls/mo)
# ════════════════════════════════════════════════════════════════════════════

def fetch_aviationstack_departures(airport_iata, limit=100):
    """
    Fetch upcoming departures from AviationStack.
    Filters to flights departing in the next 6 hours only.
    """
    if not AVIATIONSTACK_KEY:
        print(f"  ⚠ No AVIATIONSTACK_KEY — skipping")
        return pd.DataFrame()

    url = "http://api.aviationstack.com/v1/flights"
    params = {
        "access_key"    : AVIATIONSTACK_KEY,
        "dep_iata"      : airport_iata,
        "flight_status" : "scheduled",   # only upcoming flights
        "limit"         : limit,
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code == 200:
            data = r.json().get("data", [])
            if not data:
                # Fallback to active flights if no scheduled found
                params["flight_status"] = "active"
                r2   = requests.get(url, params=params, timeout=15)
                data = r2.json().get("data", []) if r2.status_code == 200 \
                       else []
            if not data:
                return pd.DataFrame()

            now = pd.Timestamp.utcnow().tz_localize(None)
            rows = []
            for f in data:
                dep = f.get("departure",{})
                arr = f.get("arrival",{})
                air = f.get("aircraft",{})
                aln = f.get("airline",{})

                # Parse departure time
                dep_sched_raw = dep.get("scheduled","")
                try:
                    dep_sched = pd.Timestamp(dep_sched_raw)
                    if dep_sched.tzinfo is not None:
                        dep_sched = dep_sched.tz_convert(None)
                except Exception:
                    continue

                # Only include flights departing in the next 6 hours
                hours_until_dep = (dep_sched - now).total_seconds() / 3600
                if hours_until_dep < 0 or hours_until_dep > 6:
                    continue

                # Parse arrival time
                arr_sched_raw = arr.get("scheduled","")
                try:
                    arr_sched = pd.Timestamp(arr_sched_raw)
                    if arr_sched.tzinfo is not None:
                        arr_sched = arr_sched.tz_convert(None)
                except Exception:
                    arr_sched = None

                rows.append({
                    "ORIGIN"            : dep.get("iata",""),
                    "DEST"              : arr.get("iata",""),
                    "flight_number"     : f.get("flight",{}).get("iata",""),
                    "Tail_Number"       : air.get("registration",""),
                    "Operating_Airline" : aln.get("iata",""),
                    "dep_scheduled"     : dep_sched,
                    "arr_scheduled"     : arr_sched,
                    "dep_delay_min"     : dep.get("delay", 0) or 0,
                    "arr_delay_min"     : arr.get("delay", 0) or 0,
                    "flight_status"     : f.get("flight_status",""),
                    "hours_until_dep"   : round(hours_until_dep, 2),
                })

            df = pd.DataFrame(rows)
            if len(df) > 0:
                print(f"  AviationStack: {len(df)} upcoming flights "
                      f"(next 6h) from {airport_iata}")
            else:
                print(f"  AviationStack: no flights in next 6h from "
                      f"{airport_iata}")
            return df
        else:
            print(f"  ⚠ AviationStack error: {r.status_code}")
            return pd.DataFrame()
    except Exception as e:
        print(f"  ⚠ AviationStack error: {e}")
        return pd.DataFrame()


# ════════════════════════════════════════════════════════════════════════════
# 4. FAA DELAY STATUS — scrapes FAA ATCSCC for ground delay programs
# ════════════════════════════════════════════════════════════════════════════

def _parse_delay_string(s):
    """Parse FAA delay strings like '1 hour 30 minutes' or '45' into minutes."""
    if not s or str(s) in ("0","None",""):
        return 0
    s   = str(s).lower()
    mins = 0
    try:
        parts = s.split()
        for i, p in enumerate(parts):
            if "hour" in p and i > 0:
                try: mins += int(parts[i-1]) * 60
                except: pass
            if "min" in p and i > 0:
                try: mins += int(parts[i-1])
                except: pass
        if mins == 0 and s.replace(".","").isdigit():
            mins = int(float(s))
    except Exception:
        pass
    return mins


def fetch_faa_avg_delays():
    """
    Fetch real-time average departure delays per airport from FAA NASSTATUS.
    Handles both JSON and XML response formats.
    Updates every 15 minutes. Free, no registration needed.
    Returns: (delays dict: iata→minutes, programs dict: iata→info)
    """
    url = "https://nasstatus.faa.gov/api/airport-status-information"
    delays, programs = {}, {}
    try:
        r = requests.get(url, timeout=10,
                         headers={"Accept":"application/json,*/*"})
        if r.status_code != 200:
            print(f"  ⚠ FAA NASSTATUS: {r.status_code}")
            return {}, {}

        text = r.text.strip()

        # JSON response
        if text.startswith(("{","[")):
            try:
                data  = r.json()
                items = data if isinstance(data, list) \
                        else data.get("data", data.get("delays", [data]))
                for item in items if isinstance(items, list) else []:
                    ap = str(item.get("arpt",
                             item.get("airport",""))).upper()
                    if ap.startswith("K") and len(ap) == 4:
                        ap = ap[1:]
                    if ap not in AIRPORTS:
                        continue
                    mins = _parse_delay_string(
                        item.get("avgDelay",
                        item.get("avg_delay",
                        item.get("delay","0"))))
                    delays[ap]   = float(mins)
                    programs[ap] = {
                        "type"         : item.get("type",""),
                        "reason"       : item.get("reason",""),
                        "avg_delay_min": float(mins),
                    }
            except Exception as e:
                print(f"  ⚠ FAA JSON parse error: {e}")

        # XML response
        elif "<" in text:
            import xml.etree.ElementTree as ET
            try:
                root = ET.fromstring(text)
                for elem in root.iter():
                    ap_elem = elem.find("ARPT")
                    if ap_elem is None:
                        ap_elem = elem.find("Arpt")
                    if ap_elem is None:
                        ap_elem = elem.find("airport")
                    if ap_elem is None or not ap_elem.text:
                        continue
                    ap = ap_elem.text.strip().upper()
                    if ap.startswith("K") and len(ap) == 4:
                        ap = ap[1:]
                    if ap not in AIRPORTS:
                        continue
                    delay_elem = (elem.find("Avg") or elem.find("AvgDelay")
                                  or elem.find("Delay") or elem.find("avg"))
                    mins = _parse_delay_string(
                        delay_elem.text if delay_elem is not None else "0")
                    delays[ap]   = float(mins)
                    programs[ap] = {
                        "type"         : elem.tag,
                        "reason"       : (elem.findtext("Reason") or ""),
                        "avg_delay_min": float(mins),
                    }
            except ET.ParseError as e:
                print(f"  ⚠ FAA XML parse error: {e}")

        if delays:
            print(f"  FAA NASSTATUS: {len(delays)} airports with active "
                  f"delays — {list(delays.keys())}")
        else:
            print(f"  FAA NASSTATUS: No active delays "
                  f"(system operating normally)")
        return delays, programs

    except Exception as e:
        print(f"  ⚠ FAA NASSTATUS error: {e}")
        return {}, {}


# ════════════════════════════════════════════════════════════════════════════
# 5. SNAPSHOT BUILDER — assembles live data into GNN-ready snapshot
# ════════════════════════════════════════════════════════════════════════════

def build_live_snapshot(snap_time, flights_df, weather_curr, weather_fore,
                         faa_delays, airport_index, static_edges,
                         hist_airport_stats, hist_tail_cumul,
                         device="cpu"):
    """
    Build a live HeteroData snapshot from API data.
    This replaces the step-5 prebuilt snapshots for real-time inference.

    snap_time         : current datetime
    flights_df        : live flights from OpenSky/AviationStack
    weather_curr      : current weather per airport
    weather_fore      : 6h forecast per airport
    faa_delays        : active FAA ground delay programs
    airport_index     : DataFrame with airport → node_idx mapping
    static_edges      : prebuilt static edge tensors
    hist_airport_stats: historical avg delays/taxi per airport (from step 5)
    hist_tail_cumul   : today's accumulated tail delays (built incrementally)
    """
    import torch
    from torch_geometric.data import HeteroData

    airports  = airport_index["airport"].tolist()
    ap2idx    = {ap: i for i, ap in enumerate(airports)}
    n_ap      = len(airports)

    data = HeteroData()
    data["airport"].snapshot_time = str(snap_time)
    data["airport"].num_nodes     = n_ap

    # ── Airport features (30 dims, matching training architecture) ───────────
    wx_curr = weather_curr.set_index("airport").to_dict("index") \
              if len(weather_curr) > 0 else {}
    wx_fore = weather_fore.set_index("airport").to_dict("index") \
              if len(weather_fore) > 0 else {}
    hist    = hist_airport_stats

    h, mo = snap_time.hour, snap_time.month
    X_ap = np.zeros((n_ap, 30), dtype=np.float32)

    for i, ap in enumerate(airports):
        s  = hist.get(ap, {})
        wc = wx_curr.get(ap, {})
        wf = wx_fore.get(ap, {})
        fd = faa_delays.get(ap, {})

        # Static (5): dims 0-4
        X_ap[i, 0] = s.get("is_hub",               0.0)
        X_ap[i, 1] = s.get("hist_avg_dep_delay",    0.0)
        X_ap[i, 2] = s.get("hist_avg_taxi_out",     0.0)
        X_ap[i, 3] = s.get("hist_avg_arr_delay",    0.0)
        X_ap[i, 4] = s.get("total_departures_norm", 0.0)

        # Current dynamic (9): dims 5-13
        faa_delay   = faa_delays.get(ap, 0.0)
        X_ap[i, 5]  = faa_delay
        X_ap[i, 6]  = 0.0
        X_ap[i, 7]  = s.get("hist_avg_taxi_out", 19.0)
        X_ap[i, 8]  = 0.0
        X_ap[i, 9]  = 0.0
        X_ap[i,10]  = 0.0
        X_ap[i,11]  = wc.get("wind_speed_ms",  0.0)
        X_ap[i,12]  = wc.get("visibility_m", 10000.0)
        X_ap[i,13]  = wc.get("precip_mm",      0.0)

        # Traffic load (6): dims 14-19 — dep/arr next 1h, 3h, 6h
        X_ap[i,14]  = 0.5   # dep_next_1h (placeholder)
        X_ap[i,15]  = 0.5   # arr_next_1h
        X_ap[i,16]  = 0.5   # dep_next_3h
        X_ap[i,17]  = 0.5   # arr_next_3h
        X_ap[i,18]  = 0.5   # dep_next_6h
        X_ap[i,19]  = 0.5   # arr_next_6h

        # Forecast weather (6): dims 20-25
        X_ap[i,20]  = wf.get("wind_3h",   0.0)
        X_ap[i,21]  = wf.get("precip_3h", 0.0)
        X_ap[i,22]  = wf.get("vis_3h", 10000.0)
        X_ap[i,23]  = wf.get("wind_6h",   0.0)
        X_ap[i,24]  = wf.get("precip_6h", 0.0)
        X_ap[i,25]  = wf.get("vis_6h", 10000.0)

        # Time embeddings (4): dims 26-29
        X_ap[i,26] = np.sin(2*np.pi*h/24)
        X_ap[i,27] = np.cos(2*np.pi*h/24)
        X_ap[i,28] = np.sin(2*np.pi*mo/12)
        X_ap[i,29] = np.cos(2*np.pi*mo/12)

    data["airport"].x = torch.tensor(X_ap, dtype=torch.float16)
    data["airport"].y = torch.zeros(n_ap, dtype=torch.float)
    data["airport"].y_mask = torch.zeros(n_ap, dtype=torch.bool)

    # ── Static edges ──────────────────────────────────────────────────────
    data["airport","congestion","airport"].edge_index = \
        static_edges["congestion_ei"]
    data["airport","congestion","airport"].edge_attr  = \
        (static_edges["congestion_ea"]
         if "congestion_ea" in static_edges
         else torch.zeros((static_edges["congestion_ei"].shape[1], 1),
                          dtype=torch.float))
    data["airport","network",   "airport"].edge_index = \
        static_edges["network_ei"]
    data["airport","network",   "airport"].edge_attr  = \
        static_edges["network_ea"]
    data["airport","rotation",  "airport"].edge_index = \
        torch.zeros((2,0), dtype=torch.long)
    data["airport","rotation",  "airport"].edge_attr  = \
        torch.zeros((0,3), dtype=torch.float)

    # ── Flight nodes from live data ───────────────────────────────────────
    if len(flights_df) > 0:
        flights_df = flights_df[
            flights_df["ORIGIN"].isin(ap2idx) &
            flights_df["DEST"].isin(ap2idx)
        ].reset_index(drop=True)

    n_fl = len(flights_df)
    MAX_DELAY = 300.0

    if n_fl > 0:
        X_fl = np.zeros((n_fl, 19), dtype=np.float32)  # +2 route stats
        t_ns = np.datetime64(snap_time, "ns").astype(np.int64)

        for i, (_, row) in enumerate(flights_df.iterrows()):
            dep_dt  = row.get("dep_scheduled") or row.get("dep_datetime")
            arr_dt  = row.get("arr_scheduled") or row.get("arr_datetime")
            tail    = str(row.get("Tail_Number",""))
            origin  = row.get("ORIGIN","")
            dep_dly = float(row.get("dep_delay_min", 0) or 0)

            # Convert dep_dt to naive UTC timestamp
            if dep_dt is not None:
                dep_ts = pd.Timestamp(dep_dt)
                if dep_ts.tzinfo is not None:
                    dep_ts = dep_ts.tz_convert(None)
            else:
                dep_ts = None

            dep_hour = dep_ts.hour if dep_ts is not None else 0
            arr_hour = pd.Timestamp(arr_dt).hour \
                       if arr_dt is not None and pd.notna(pd.Timestamp(arr_dt)) \
                       else 0
            dow = dep_ts.dayofweek if dep_ts is not None else 0

            # Real time_to_dep from actual departure time
            if dep_ts is not None:
                snap_naive = snap_time.replace(tzinfo=None) \
                             if hasattr(snap_time, 'tzinfo') and \
                             snap_time.tzinfo is not None \
                             else snap_time
                h2dep = max(0.0, (dep_ts - pd.Timestamp(snap_naive)
                                  ).total_seconds() / 3600)
            else:
                h2dep = row.get("hours_until_dep", 1.0)

            tail_info = hist_tail_cumul.get(tail, {})
            cumul     = tail_info.get("cumul", 0.0)
            legs      = tail_info.get("legs",  0)
            immed     = tail_info.get("immed", dep_dly)

            X_fl[i, 1]  = np.sin(2*np.pi*dep_hour/24)
            X_fl[i, 2]  = np.cos(2*np.pi*dep_hour/24)
            X_fl[i, 5]  = 0.3
            X_fl[i, 6]  = 1.0 if (cumul == 0 and immed == 0) else 0.0
            X_fl[i, 8]  = np.sin(2*np.pi*dow/7)
            X_fl[i, 9]  = np.cos(2*np.pi*dow/7)
            X_fl[i,10]  = 1.0 if origin in {
                "ATL","ORD","DFW","DEN","LAX","JFK","SFO",
                "MIA","BOS","EWR","LGA","IAH","IAD","CLT"} else 0.0
            X_fl[i,12]  = np.sin(2*np.pi*arr_hour/24)
            X_fl[i,13]  = np.cos(2*np.pi*arr_hour/24)

            time_to_dep = min(h2dep/24.0, 1.0)
            X_fl[i,14]  = time_to_dep
            X_fl[i,15]  = np.clip(cumul/300.0, 0, 1)
            X_fl[i,16]  = min(legs/6.0, 1.0)
            # Route stats (17,18) — zero here, populated when route_stats loaded
            X_fl[i,17]  = 0.0   # hist_route_avg — loaded via route_stats.parquet
            X_fl[i,18]  = 0.0   # hist_route_std — loaded via route_stats.parquet

            # ── Horizon-aware gate feature masking ───────────────────────────
            # Matches 06_train_gnn.py masking exactly:
            #   < 1h  : inbound delay known (plane landed), dep/taxi unknown
            #   1-2h  : dep_delay + taxi zero (gate not ready)
            #   > 2h  : all gate features zero (no info yet)
            FULL_MASK    = 2.0 / 24
            PARTIAL_MASK = 1.0 / 24

            if time_to_dep < PARTIAL_MASK:
                # < 1h: inbound delay is known from tail tracker
                X_fl[i, 0]  = 0.0   # dep_delay: not yet recorded
                X_fl[i, 3]  = 0.0   # turnaround: not completed
                X_fl[i, 4]  = np.clip(immed/MAX_DELAY, 0, 1)  # inbound known
                X_fl[i, 7]  = 0.0   # taxi_out: not started
                X_fl[i,11]  = 0.0   # carrier: unknown

            elif time_to_dep < FULL_MASK:
                # 1-2h: dep_delay and taxi unknown, inbound may be known
                X_fl[i, 0]  = 0.0
                X_fl[i, 3]  = 0.0
                X_fl[i, 4]  = np.clip(immed/MAX_DELAY, 0, 1)
                X_fl[i, 7]  = 0.0
                X_fl[i,11]  = 0.0

            else:
                # > 2h: all gate features zero (already 0 from np.zeros)
                pass

        data["flight"].x         = torch.tensor(X_fl, dtype=torch.float16)
        data["flight"].num_nodes = n_fl

        # Store metadata for display
        data["flight"].ORIGIN  = flights_df["ORIGIN"].values
        data["flight"].DEST    = flights_df["DEST"].values
        data["flight"].tail    = flights_df.get("Tail_Number",
                                  pd.Series([""] * n_fl)).values
        data["flight"].airline = flights_df.get("Operating_Airline",
                                  pd.Series([""] * n_fl)).values

        # Causal edges
        dep_src, dep_dst, arr_src, arr_dst = [], [], [], []
        for i, (_, row) in enumerate(flights_df.iterrows()):
            o = row.get("ORIGIN",""); d = row.get("DEST","")
            t = str(row.get("Tail_Number",""))
            if o in ap2idx:
                dep_src.append(i); dep_dst.append(ap2idx[o])
            if d in ap2idx:
                ti = hist_tail_cumul.get(t, {})
                if ti.get("cumul", 0) > 15 or ti.get("immed", 0) > 15:
                    arr_src.append(i); arr_dst.append(ap2idx[d])

        def mk(s, d):
            if not s:
                return (torch.zeros((2,0),dtype=torch.long),
                        torch.ones((0,1),dtype=torch.float))
            return (torch.tensor([s,d],dtype=torch.long),
                    torch.ones((len(s),1),dtype=torch.float))

        de_ei, de_ea = mk(dep_src, dep_dst)
        ar_ei, ar_ea = mk(arr_src, arr_dst)
        data["flight","departs_from","airport"].edge_index = de_ei
        data["flight","departs_from","airport"].edge_attr  = de_ea
        data["flight","arrives_at",  "airport"].edge_index = ar_ei
        data["flight","arrives_at",  "airport"].edge_attr  = ar_ea
        data["flight","rotation",    "flight" ].edge_index = \
            torch.zeros((2,0),dtype=torch.long)
        data["flight","rotation",    "flight" ].edge_attr  = \
            torch.zeros((0,2),dtype=torch.float)

    else:
        data["flight"].x         = torch.zeros((0,17),dtype=torch.float16)
        data["flight"].num_nodes = 0
        for et in [("flight","departs_from","airport"),
                   ("flight","arrives_at",  "airport"),
                   ("flight","rotation",    "flight")]:
            d = 1 if et[1] != "rotation" else 2
            data[et].edge_index = torch.zeros((2,0),dtype=torch.long)
            data[et].edge_attr  = torch.zeros((0,d),dtype=torch.float)

    data["flight"].y      = torch.zeros(n_fl, dtype=torch.float)
    data["flight"].y_mask = torch.zeros(n_fl, dtype=torch.bool)

    return data, flights_df


# ════════════════════════════════════════════════════════════════════════════
# 6. TAIL CUMULATIVE DELAY TRACKER — updates as flights land
# ════════════════════════════════════════════════════════════════════════════

class TailDelayTracker:
    """
    Maintains real-time cumulative delay per tail number.
    Updated each cycle as AviationStack reports actual arrival times.
    """
    def __init__(self):
        self.today    = datetime.now(timezone.utc).date()
        self.tail_map = {}   # tail → {cumul, legs, last_arr_delay}

    def update(self, flights_df):
        """Update tracker with latest flight data."""
        today = datetime.now(timezone.utc).date()
        if today != self.today:
            self.tail_map = {}   # reset at midnight
            self.today    = today

        for _, row in flights_df.iterrows():
            tail   = str(row.get("Tail_Number",""))
            status = str(row.get("flight_status",""))
            delay  = float(row.get("arr_delay_min", 0) or 0)

            if not tail or tail == "nan":
                continue

            if status == "landed" and delay > 0:
                if tail not in self.tail_map:
                    self.tail_map[tail] = {"cumul":0.0,"legs":0,"immed":0.0}
                self.tail_map[tail]["cumul"] += max(0.0, delay)
                self.tail_map[tail]["legs"]  += 1
                self.tail_map[tail]["immed"]  = delay

    def get_lookup(self):
        return self.tail_map


# ════════════════════════════════════════════════════════════════════════════
# 7. MAIN REAL-TIME LOOP
# ════════════════════════════════════════════════════════════════════════════

def load_hist_airport_stats():
    """Load historical airport statistics for feature building."""
    idx_path = os.path.join(GRAPH_DATA_DIR, "airport_index.parquet")
    ap_idx = pd.read_parquet(idx_path) if os.path.exists(idx_path) \
             else pd.DataFrame({"airport": sorted(AIRPORTS.keys()),
                                "node_idx": range(len(AIRPORTS))})

    stats = {}

    # Try weather_node_features.parquet first (has DepDelay/ArrDelay/TaxiOut)
    wx_path = os.path.join(GRAPH_DATA_DIR, "weather_node_features.parquet")
    raw_path = os.path.join(BASE_DIR, "flights_2018_2022.parquet")

    source_path = None
    if os.path.exists(wx_path):
        source_path = wx_path
    elif os.path.exists(raw_path):
        source_path = raw_path

    if source_path:
        print(f"  Loading hist stats from {os.path.basename(source_path)} ...")
        fl = pd.read_parquet(source_path,
                             columns=["ORIGIN","DEST","DepDelay",
                                      "ArrDelay","TaxiOut"])
        fl = fl[fl["ORIGIN"].isin(AIRPORTS) & fl["DEST"].isin(AIRPORTS)]

        dep_s = fl.groupby("ORIGIN").agg(
            hist_avg_dep_delay=("DepDelay","mean"),
            hist_avg_taxi_out =("TaxiOut", "mean"),
            total_departures  =("DepDelay","count")
        ).reset_index()
        arr_s = fl.groupby("DEST")["ArrDelay"].mean().reset_index()

        max_dep = dep_s["total_departures"].max()
        hub_set = set(dep_s.nlargest(30,"total_departures")["ORIGIN"])

        for _, row in dep_s.iterrows():
            ap = row["ORIGIN"]
            stats[ap] = {
                "is_hub"               : 1.0 if ap in hub_set else 0.0,
                "hist_avg_dep_delay"   : float(row["hist_avg_dep_delay"] or 0),
                "hist_avg_taxi_out"    : float(row["hist_avg_taxi_out"]  or 19),
                "hist_avg_arr_delay"   : 0.0,
                "total_departures_norm": float(row["total_departures"]) / max_dep,
            }
        for _, row in arr_s.iterrows():
            ap = row["DEST"]
            if ap in stats:
                stats[ap]["hist_avg_arr_delay"] = float(row["ArrDelay"] or 0)

        print(f"  ✓ Historical stats for {len(stats)} airports")
    else:
        # Fallback — use hardcoded reasonable defaults
        print("  ⚠ No source parquet found — using default airport stats")
        hub_set = {"ATL","ORD","DFW","DEN","LAX","JFK","SFO",
                   "MIA","BOS","EWR","LGA","IAH","IAD","CLT",
                   "PHX","SEA","MSP","DTW","PHL","SLC"}
        for ap in AIRPORTS:
            stats[ap] = {
                "is_hub"               : 1.0 if ap in hub_set else 0.0,
                "hist_avg_dep_delay"   : 8.0,
                "hist_avg_taxi_out"    : 19.0,
                "hist_avg_arr_delay"   : 5.0,
                "total_departures_norm": 0.5,
            }

    return ap_idx, stats


def run_realtime_predict(model, static_edges, airport_index,
                          hist_stats, tail_tracker, device):
    """One cycle: fetch data → build snapshot → run model → return predictions."""
    snap_time = datetime.now(timezone.utc)
    print(f"\n{'─'*55}")
    print(f"Real-time prediction cycle: {snap_time.strftime('%Y-%m-%d %H:%M UTC')}")

    # Fetch live data
    weather_curr, weather_fore = fetch_all_weather()
    faa_delays, faa_programs      = fetch_faa_avg_delays()

    # Fetch flights — use AviationStack if key available, else OpenSky
    if AVIATIONSTACK_KEY:
        flights_list = []
        for ap in list(AIRPORTS.keys())[:10]:   # conserve free tier calls
            df = fetch_aviationstack_departures(ap, limit=50)
            if len(df) > 0:
                flights_list.append(df)
            time.sleep(1)
        flights_df = pd.concat(flights_list, ignore_index=True) \
                     if flights_list else pd.DataFrame()
    else:
        flights_df = pd.DataFrame()
        print("  No AviationStack key — using airport-only predictions")
        print("  Set AVIATIONSTACK_KEY in .env for full flight predictions")

    # Update tail cumulative delay tracker
    if len(flights_df) > 0:
        tail_tracker.update(flights_df)

    # Build snapshot
    snap, flights_df = build_live_snapshot(
        snap_time, flights_df, weather_curr, weather_fore,
        faa_delays, airport_index, static_edges,
        hist_stats, tail_tracker.get_lookup(), device)

    # Run model
    import torch
    airports = airport_index["airport"].tolist()
    ap_h     = model.init_hidden(torch.device(device))

    with torch.no_grad():
        snap = snap.to(device)
        ap_pred, fl_pred, ap_h = model(snap, ap_h)

    # Format airport predictions
    ap_preds = ap_pred.cpu().numpy()
    ap_results = pd.DataFrame({
        "airport"   : airports,
        "pred_delay": np.round(ap_preds, 1),
        "lat"       : [AIRPORT_COORDS[ap][0] for ap in airports],
        "lon"       : [AIRPORT_COORDS[ap][1] for ap in airports],
    })

    # Format flight predictions
    n_fl = snap["flight"].num_nodes
    if n_fl > 0 and len(fl_pred) > 0:
        fl_preds = fl_pred.cpu().numpy()

        dep_ei = snap["flight","departs_from","airport"].edge_index
        arr_ei = snap["flight","arrives_at",  "airport"].edge_index
        dep_active = set(dep_ei[0].cpu().numpy().tolist()) \
                     if dep_ei.shape[1] > 0 else set()
        arr_active = set(arr_ei[0].cpu().numpy().tolist()) \
                     if arr_ei.shape[1] > 0 else set()

        fl_results = []
        for i in range(min(n_fl, len(fl_preds))):
            if i in dep_active and i in arr_active:
                cause = "Congestion + Inbound delay"
            elif i in dep_active:
                cause = "Airport congestion"
            elif i in arr_active:
                cause = "Inbound aircraft delay"
            else:
                cause = "Schedule/Weather pattern"

            pred = float(fl_preds[i])
            fl_results.append({
                "ORIGIN"      : snap["flight"].ORIGIN[i]
                                 if hasattr(snap["flight"],"ORIGIN") else "",
                "DEST"        : snap["flight"].DEST[i]
                                 if hasattr(snap["flight"],"DEST")   else "",
                "Tail_Number" : snap["flight"].tail[i]
                                 if hasattr(snap["flight"],"tail")   else "",
                "Airline"     : snap["flight"].airline[i]
                                 if hasattr(snap["flight"],"airline") else "",
                "pred_delay"  : round(pred, 1),
                "delay_cause" : cause,
                "status"      : "🟢 On Time" if pred < 5
                                 else ("🟡 Minor" if pred < 20
                                       else ("🟠 Moderate" if pred < 45
                                             else "🔴 Severe")),
                "snapshot_utc": snap_time.isoformat(),
            })
        fl_df = pd.DataFrame(fl_results)
    else:
        fl_df = pd.DataFrame()

    # Print summary
    n_congested = (ap_results["pred_delay"] > 15).sum()
    worst = ap_results.loc[ap_results["pred_delay"].idxmax()]
    print(f"\n  Airport predictions:")
    print(f"    Congested airports (>15min): {n_congested}")
    print(f"    Worst: {worst['airport']} — {worst['pred_delay']:.0f} min")
    if len(fl_df) > 0:
        n_del = (fl_df["pred_delay"] > 15).sum()
        print(f"  Flight predictions: {len(fl_df)} flights, "
              f"{n_del} predicted delayed")

    return ap_results, fl_df, snap_time


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["predict","dashboard","test"],
                        default="predict")
    args = parser.parse_args()

    print("=" * 55)
    print("STEP 8 — REAL-TIME API CONNECTOR")
    print("=" * 55)

    if args.mode == "test":
        print("\nTesting API connections ...")
        print("\n1. OpenSky Network:")
        states = fetch_opensky_states()
        print(f"   {'✅' if len(states)>0 else '❌'} "
              f"{len(states)} aircraft tracked")

        print("\n2. NWS Weather API:")
        wx = fetch_nws_current("ATL")
        print(f"   {'✅' if wx['wind_speed_ms'] >= 0 else '❌'} "
              f"ATL wind: {wx['wind_speed_ms']:.1f} m/s, "
              f"vis: {wx['visibility_m']:.0f}m")

        print("\n3. FAA Delay Status:")
        delays, _ = fetch_faa_avg_delays()
        print(f"   ✅ {len(delays)} airports with active programs")

        print("\n4. AviationStack:")
        if AVIATIONSTACK_KEY:
            df = fetch_aviationstack_departures("ATL", limit=5)
            print(f"   {'✅' if len(df)>0 else '❌'} "
                  f"{len(df)} flights returned")
        else:
            print("   ⚠ No AVIATIONSTACK_KEY in environment")
            print("   Get a free key at https://aviationstack.com")
            print("   Add to .env file: AVIATIONSTACK_KEY=your_key_here")
        return

    # Load model
    import torch
    ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint not found: {ckpt_path}")
        return

    # Import model class from step 7
    import sys
    sys.path.insert(0, BASE_DIR)
    from importlib import import_module

    # Load checkpoint
    ckpt   = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    device = "cpu"

    # Lazy import model
    from torch_geometric.nn import HGTConv, Linear
    import torch.nn as nn

    # Re-use FlightDelayGNN from 07_dashboard.py
    sys.path.insert(0, BASE_DIR)
    spec = import_module("07_dashboard") if "07_dashboard" in sys.modules \
           else None

    print("\nLoading supporting files ...")
    static_edges  = torch.load(
        os.path.join(GRAPH_DATA_DIR, "static_edges.pt"),
        map_location="cpu", weights_only=False)
    airport_index, hist_stats = load_hist_airport_stats()
    tail_tracker = TailDelayTracker()

    print(f"  ✓ {len(airport_index)} airports")
    print(f"  ✓ Historical stats for {len(hist_stats)} airports")
    print(f"  ✓ AviationStack: {'enabled' if AVIATIONSTACK_KEY else 'disabled (no key)'}")

    if args.mode == "predict":
        # Load model properly
        from importlib.util import spec_from_file_location, module_from_spec
        dashboard_path = os.path.join(BASE_DIR, "07_dashboard.py")
        spec_mod = spec_from_file_location("dashboard", dashboard_path)
        mod      = module_from_spec(spec_mod)
        spec_mod.loader.exec_module(mod)
        model = mod.load_model(ckpt_path, device)

        ap_df, fl_df, ts = run_realtime_predict(
            model, static_edges, airport_index,
            hist_stats, tail_tracker, device)

        # Save results
        out_dir = os.path.join(BASE_DIR, "outputs")
        os.makedirs(out_dir, exist_ok=True)
        ts_str = ts.strftime("%Y%m%d_%H%M")
        ap_df.to_csv(os.path.join(out_dir, f"airport_pred_{ts_str}.csv"),
                     index=False)
        if len(fl_df) > 0:
            fl_df.to_csv(os.path.join(out_dir, f"flight_pred_{ts_str}.csv"),
                         index=False)
        print(f"\n  ✅ Results saved to outputs/")

    elif args.mode == "dashboard":
        print("\nReal-time dashboard mode — refreshes every "
              f"{REFRESH_MINUTES} minutes")
        print("Open http://127.0.0.1:8051 in your browser\n")
        # Launch dashboard with live data refresh
        # TODO: integrate with 07_dashboard.py live mode
        print("(Full live dashboard integration coming in next update)")
        print("For now: run --mode predict periodically and use 07_dashboard.py")


if __name__ == "__main__":
    main()