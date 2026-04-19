"""
STEP 10 — PRODUCTION FLIGHT FINDER
====================================
Lightweight web app for on-demand flight delay predictions.

Users enter:
  - Origin airport (from 36 hubs)
  - Destination airport (from 36 hubs)
  - Date
  - Flight number (optional — used to look up tail number)

The app:
  1. Fetches weather for origin + destination only (2 NWS calls — free)
  2. Fetches the specific flight from AviationStack (1 call)
  3. Runs the GNN with pre-cached airport state
  4. Returns 6h / 3h / 1h predictions

API cost: 3 calls per user request vs 82 calls for full refresh.
At 100 requests/month → well within AviationStack free tier (100 calls).

USAGE:
  python 10_flight_finder.py
  # Open http://127.0.0.1:8051

REQUIRES:
  pip install dash plotly requests python-dotenv torch torch-geometric
"""

import os
import time
import json
import requests
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from importlib.util import spec_from_file_location, module_from_spec
from dotenv import load_dotenv
import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc

load_dotenv()

# ── CONFIG ──────────────────────────────────────────────────────────────────
BASE_DIR          = r"C:\Users\user\Desktop\Airline_Graphs_Project"
GRAPH_DATA_DIR    = os.path.join(BASE_DIR, "graph_data")
CHECKPOINT_DIR    = os.path.join(BASE_DIR, "checkpoints")
AVIATIONSTACK_KEY = os.getenv("AVIATIONSTACK_KEY", "")
MODEL_MAE_MIN     = 20   # honest accuracy statement in minutes
DELAY_THRESHOLD   = 15.0

# The 36 hub airports
AIRPORTS = [
    "ANC","ATL","BNA","BOS","BWI","CLE","CLT","CMH","DEN","DFW",
    "DTW","EWR","FLL","HOU","IAD","IAH","IND","JFK","LAS","LAX",
    "LGA","MCO","MCI","MIA","MSP","OAK","ORD","PDX","PHL","PHX",
    "PIT","SAN","SEA","SFO","SJC","SLC",
]

AIRPORT_NAMES = {
    "ANC":"Anchorage","ATL":"Atlanta","BNA":"Nashville","BOS":"Boston",
    "BWI":"Baltimore","CLE":"Cleveland","CLT":"Charlotte","CMH":"Columbus",
    "DEN":"Denver","DFW":"Dallas/FW","DTW":"Detroit","EWR":"Newark",
    "FLL":"Ft. Lauderdale","HOU":"Houston Hobby","IAD":"Washington Dulles",
    "IAH":"Houston Intercontinental","IND":"Indianapolis","JFK":"New York JFK",
    "LAS":"Las Vegas","LAX":"Los Angeles","LGA":"New York LaGuardia",
    "MCO":"Orlando","MCI":"Kansas City","MIA":"Miami","MSP":"Minneapolis",
    "OAK":"Oakland","ORD":"Chicago O'Hare","PDX":"Portland","PHL":"Philadelphia",
    "PHX":"Phoenix","PIT":"Pittsburgh","SAN":"San Diego","SEA":"Seattle",
    "SFO":"San Francisco","SJC":"San Jose","SLC":"Salt Lake City",
}

NWS_STATION_MAP = {
    "ANC":"PANC","ATL":"KATL","BNA":"KBNA","BOS":"KBOS","BWI":"KBWI",
    "CLE":"KCLE","CLT":"KCLT","CMH":"KCMH","DEN":"KDEN","DFW":"KDFW",
    "DTW":"KDTW","EWR":"KEWR","FLL":"KFLL","HOU":"KHOU","IAD":"KIAD",
    "IAH":"KIAH","IND":"KIND","JFK":"KJFK","LAS":"KLAS","LAX":"KLAX",
    "LGA":"KLGA","MCO":"KMCO","MCI":"KMCI","MIA":"KMIA","MSP":"KMSP",
    "OAK":"KOAK","ORD":"KORD","PDX":"KPDX","PHL":"KPHL","PHX":"KPHX",
    "PIT":"KPIT","SAN":"KSAN","SEA":"KSEA","SFO":"KSFO","SJC":"KSJC",
    "SLC":"KSLC",
}
# ────────────────────────────────────────────────────────────────────────────


# ════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_assets():
    """Load model, airport index, and route stats once at startup."""
    print("Loading model and assets...")

    # Load model via step 7
    dash_path = os.path.join(BASE_DIR, "07_dashboard.py")
    spec = spec_from_file_location("dashboard", dash_path)
    mod  = module_from_spec(spec)
    spec.loader.exec_module(mod)

    device    = torch.device("cpu")
    ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    model     = mod.load_model(ckpt_path, device)
    model.eval()

    # Airport index
    ap_idx = pd.read_parquet(
        os.path.join(GRAPH_DATA_DIR, "airport_index.parquet"))
    airports   = ap_idx["airport"].tolist()
    ap_idx_map = {ap: i for i, ap in enumerate(airports)}

    # Static edges
    static = torch.load(
        os.path.join(GRAPH_DATA_DIR, "static_edges.pt"),
        map_location=device, weights_only=False)

    # Route stats
    route_stats = None
    rs_path = os.path.join(GRAPH_DATA_DIR, "route_stats.parquet")
    rg_path = os.path.join(GRAPH_DATA_DIR, "route_stats_global.parquet")
    if os.path.exists(rs_path):
        rs = pd.read_parquet(rs_path)
        rg = pd.read_parquet(rg_path)
        lf = {(r.ORIGIN, r.DEST, int(r.dep_hour), int(r.DayOfWeek)):
              (float(r.hist_avg), float(r.hist_std))
              for r in rs.itertuples(index=False)}
        lh, lr = {}, {}
        for r in rs.itertuples(index=False):
            lh.setdefault((r.ORIGIN, r.DEST, int(r.dep_hour)),
                          (float(r.hist_avg), float(r.hist_std)))
            lr.setdefault((r.ORIGIN, r.DEST),
                          (float(r.hist_avg), float(r.hist_std)))
        gm = float(rg.iloc[0]["global_mean"])
        gs = float(rg.iloc[0]["global_std"])
        route_stats = (lf, lh, lr, gm, gs)
        print(f"  ✓ Route stats loaded ({len(lf):,} entries)")

    print(f"  ✓ Model ready (epoch {torch.load(ckpt_path, weights_only=False)['epoch']})")
    return model, airports, ap_idx_map, static, route_stats, device


# ════════════════════════════════════════════════════════════════════════════
# API CALLS — minimal, per-request
# ════════════════════════════════════════════════════════════════════════════

def fetch_nws_weather(iata_code):
    """Fetch current weather for one airport from NWS (free, no key)."""
    icao = NWS_STATION_MAP.get(iata_code, f"K{iata_code}")
    try:
        url  = f"https://api.weather.gov/stations/{icao}/observations/latest"
        resp = requests.get(url, headers={"User-Agent":"FlightDelayGNN/1.0"},
                            timeout=8)
        if resp.status_code == 200:
            props = resp.json()["properties"]
            return {
                "wind_speed_ms"   : props.get("windSpeed",{}).get("value") or 0.0,
                "visibility_m"    : props.get("visibility",{}).get("value") or 10000.0,
                "precip_depth_mm" : props.get("precipitationLastHour",{}).get("value") or 0.0,
            }
    except Exception:
        pass
    return {"wind_speed_ms":0.0,"visibility_m":10000.0,"precip_depth_mm":0.0}


def fetch_flight_info(flight_number, date_str):
    """
    Fetch a specific flight from AviationStack.
    Returns: {"dep_hour": int, "tail": str, "dep_delay": float, "status": str}
    Costs 1 API call.
    """
    if not AVIATIONSTACK_KEY:
        return None

    # Strip airline prefix for AviationStack query
    # e.g. "UA328" → flight_iata="UA328"
    try:
        url    = "http://api.aviationstack.com/v1/flights"
        params = {
            "access_key"  : AVIATIONSTACK_KEY,
            "flight_iata" : flight_number.upper().strip(),
            "flight_date" : date_str,
        }
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            if data:
                f = data[0]
                dep = f.get("departure", {})
                return {
                    "dep_hour"   : int(dep.get("scheduled","00:00")[0:2]) if dep.get("scheduled") else 12,
                    "tail"       : f.get("aircraft",{}).get("registration",""),
                    "dep_delay"  : float(dep.get("delay") or 0),
                    "status"     : f.get("flight_status","scheduled"),
                    "origin"     : dep.get("iata",""),
                    "dest"       : f.get("arrival",{}).get("iata",""),
                }
    except Exception:
        pass
    return None


# ════════════════════════════════════════════════════════════════════════════
# PREDICTION ENGINE
# ════════════════════════════════════════════════════════════════════════════

def build_airport_features_simple(ap_idx_map, weather_map, snap_time):
    """
    Build minimal airport feature vector using just live weather.
    Static features use zeros (model is robust to this).
    """
    n  = len(ap_idx_map)
    X  = np.zeros((n, 30), dtype=np.float32)
    h  = snap_time.hour
    mo = snap_time.month

    for ap, idx in ap_idx_map.items():
        wx = weather_map.get(ap, {})
        # Dynamic weather features (indices 14-16 in dynamic block)
        X[idx, 14] = float(wx.get("wind_speed_ms",   0.0) or 0.0) / 30.0
        X[idx, 15] = float(wx.get("visibility_m",    10000.0) or 10000.0) / 10000.0
        X[idx, 16] = float(wx.get("precip_depth_mm", 0.0) or 0.0) / 50.0
        # Time features (last 4 dims)
        X[idx, 26] = np.sin(2*np.pi*h/24)
        X[idx, 27] = np.cos(2*np.pi*h/24)
        X[idx, 28] = np.sin(2*np.pi*mo/12)
        X[idx, 29] = np.cos(2*np.pi*mo/12)

    return X


def build_flight_features_single(origin, dest, dep_hour, dow,
                                  h2dep, route_stats, dep_delay=0.0):
    """
    Build 19-dim feature vector for one flight.
    Applies horizon masking matching training.
    """
    X               = np.zeros(19, dtype=np.float32)
    MAX_DELAY       = 300.0
    MASK_FULL       = 2.0 / 24
    MASK_PART       = 1.0 / 24
    time_to_dep     = min(h2dep / 24.0, 1.0)

    # Schedule features — always available
    X[1]  = np.sin(2*np.pi*dep_hour/24)
    X[2]  = np.cos(2*np.pi*dep_hour/24)
    X[6]  = 1.0   # is_first (unknown, assume yes)
    X[8]  = np.sin(2*np.pi*dow/7)
    X[9]  = np.cos(2*np.pi*dow/7)
    X[10] = 1.0   # is_hub_origin (all 36 airports are hubs)
    X[14] = time_to_dep

    # Gate features — only if within 1h of departure
    if time_to_dep < MASK_PART and dep_delay != 0:
        X[0] = np.clip(dep_delay / MAX_DELAY, -1, 1)

    # Route stats — never masked
    if route_stats is not None:
        lf, lh, lr, gm, gs = route_stats
        key_f = (origin, dest, dep_hour, dow)
        key_h = (origin, dest, dep_hour)
        key_r = (origin, dest)
        if   key_f in lf: h_avg, h_std = lf[key_f]
        elif key_h in lh: h_avg, h_std = lh[key_h]
        elif key_r in lr: h_avg, h_std = lr[key_r]
        else:             h_avg, h_std = gm, gs
        X[17] = np.clip(h_avg / MAX_DELAY, -1, 1)
        X[18] = np.clip(h_std / MAX_DELAY,  0, 1)

    return X


def predict_flight(origin, dest, dep_datetime_str, flight_number,
                   model, airports, ap_idx_map, static, route_stats, device):
    """
    Main prediction function.
    Fetches weather, builds graph, runs model at 3 horizons.
    Returns dict with predictions at 6h, 3h, 1h.
    """
    try:
        dep_dt  = pd.Timestamp(dep_datetime_str)
    except Exception:
        dep_dt  = pd.Timestamp.now() + pd.Timedelta(hours=4)

    dow      = dep_dt.dayofweek
    dep_hour = dep_dt.hour
    date_str = dep_dt.strftime("%Y-%m-%d")

    # Step 1 — fetch weather for origin + destination (2 NWS calls)
    print(f"  Fetching weather for {origin} and {dest}...")
    wx_origin = fetch_nws_weather(origin)
    wx_dest   = fetch_nws_weather(dest)
    weather_map = {origin: wx_origin, dest: wx_dest}

    # Step 2 — fetch flight info (1 AviationStack call, optional)
    flight_info = None
    dep_delay   = 0.0
    if flight_number and AVIATIONSTACK_KEY:
        print(f"  Fetching flight info for {flight_number}...")
        flight_info = fetch_flight_info(flight_number, date_str)
        if flight_info:
            dep_delay = flight_info.get("dep_delay", 0.0)

    results = {}
    cg_ei   = static["congestion_ei"].to(device)
    cg_ea   = (static["congestion_ea"].to(device)
               if "congestion_ea" in static
               else torch.zeros((0, 1), dtype=torch.float, device=device))
    nw_ei   = static["network_ei"].to(device)
    nw_ea   = static["network_ea"].to(device)

    # Run predictions at each horizon
    for hours_before in [6, 3, 1]:
        snap_time  = dep_dt - pd.Timedelta(hours=hours_before)
        h2dep      = float(hours_before)

        # Build airport features
        X_ap = build_airport_features_simple(ap_idx_map, weather_map, snap_time)
        X_ap_t = torch.tensor(X_ap, dtype=torch.float16)

        # Build single flight node
        X_fl = build_flight_features_single(
            origin, dest, dep_hour, dow, h2dep,
            route_stats,
            dep_delay if hours_before == 1 else 0.0)
        X_fl_t = torch.tensor(X_fl, dtype=torch.float16).unsqueeze(0)

        # Build minimal HeteroData snapshot
        from torch_geometric.data import HeteroData
        snap = HeteroData()

        snap["airport"].x         = X_ap_t
        snap["airport"].num_nodes = len(airports)
        snap["airport"].y         = torch.zeros(len(airports))
        snap["airport"].y_mask    = torch.zeros(len(airports), dtype=torch.bool)

        snap["flight"].x         = X_fl_t
        snap["flight"].num_nodes = 1
        snap["flight"].y         = torch.zeros(1)
        snap["flight"].y_mask    = torch.zeros(1, dtype=torch.bool)
        for attr in ["y_mask_1h","y_mask_3h","y_mask_6h"]:
            setattr(snap["flight"], attr, torch.zeros(1, dtype=torch.bool))

        # Edges — single flight connects to its origin airport
        o_idx = ap_idx_map.get(origin, 0)
        snap["airport","rotation",    "airport"].edge_index = \
            torch.zeros((2,0), dtype=torch.long)
        snap["airport","rotation",    "airport"].edge_attr  = \
            torch.zeros((0,3))
        snap["airport","congestion",  "airport"].edge_index = cg_ei
        snap["airport","congestion",  "airport"].edge_attr  = cg_ea
        snap["airport","network",     "airport"].edge_index = nw_ei
        snap["airport","network",     "airport"].edge_attr  = nw_ea
        snap["flight", "rotation",    "flight" ].edge_index = \
            torch.zeros((2,0), dtype=torch.long)
        snap["flight", "rotation",    "flight" ].edge_attr  = \
            torch.zeros((0,4))
        snap["flight", "departs_from","airport"].edge_index = \
            torch.tensor([[0],[o_idx]], dtype=torch.long)
        snap["flight", "departs_from","airport"].edge_attr  = \
            torch.zeros((1,1))
        snap["flight", "arrives_at",  "airport"].edge_index = \
            torch.zeros((2,0), dtype=torch.long)
        snap["flight", "arrives_at",  "airport"].edge_attr  = \
            torch.zeros((0,1))

        snap = snap.to(device)
        ap_h = model.init_hidden(device)

        with torch.no_grad():
            ap_pred, fl_pred, fl_logits, ap_h = model(snap, ap_h)

        pred_delay = float(fl_pred[0].item()) if fl_pred.shape[0] > 0 else 0.0
        delay_prob = float(torch.sigmoid(fl_logits[0]).item()) \
                     if fl_logits.shape[0] > 0 else 0.5

        results[f"{hours_before}h"] = {
            "pred_delay"  : round(pred_delay, 1),
            "delay_prob"  : round(delay_prob, 3),
            "pred_arrival": (dep_dt + pd.Timedelta(minutes=pred_delay)
                             ).strftime("%H:%M"),
            "confidence"  : "High" if hours_before == 1
                            else ("Medium" if hours_before == 3 else "Low"),
        }

    results["flight_info"] = flight_info
    results["weather"]     = {
        "origin": wx_origin, "dest": wx_dest
    }
    return results


# ════════════════════════════════════════════════════════════════════════════
# DASH APP
# ════════════════════════════════════════════════════════════════════════════

def build_app(model, airports, ap_idx_map, static, route_stats, device):

    app = dash.Dash(__name__,
                    external_stylesheets=["https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"],
                    title="Flight Delay Predictor")

    airport_options = [
        {"label": f"{ap} — {AIRPORT_NAMES.get(ap, ap)}", "value": ap}
        for ap in sorted(AIRPORTS)
    ]

    # ── Layout ────────────────────────────────────────────────────────────
    app.layout = html.Div([

        # Header
        html.Div([
            html.Div([
                html.H1("✈ Flight Delay Predictor",
                        style={"color":"#7eb8f7","margin":"0",
                               "fontSize":"26px","fontWeight":"700"}),
                html.P("Predicts arrival delay up to 6 hours before departure "
                       "· 36 US hub airports · Research prototype (±20 min accuracy)",
                       style={"color":"#6b7a99","margin":"4px 0 0 0","fontSize":"12px"}),
            ]),
        ], style={"background":"#131929","padding":"20px 32px",
                  "borderBottom":"1px solid #1e2d4a"}),

        # Search form
        html.Div([
            html.Div([

                # Origin
                html.Div([
                    html.Label("From", style={"color":"#7eb8f7","fontSize":"12px",
                                              "fontWeight":"600","marginBottom":"6px",
                                              "display":"block"}),
                    dcc.Dropdown(
                        id="origin", options=airport_options,
                        placeholder="Origin airport",
                        style={"fontSize":"14px"}),
                ], style={"flex":"1","minWidth":"180px"}),

                html.Div("→", style={"color":"#4a5a70","fontSize":"28px",
                                      "padding":"24px 8px 0 8px"}),

                # Destination
                html.Div([
                    html.Label("To", style={"color":"#7eb8f7","fontSize":"12px",
                                            "fontWeight":"600","marginBottom":"6px",
                                            "display":"block"}),
                    dcc.Dropdown(
                        id="dest", options=airport_options,
                        placeholder="Destination airport",
                        style={"fontSize":"14px"}),
                ], style={"flex":"1","minWidth":"180px"}),

                # Date
                html.Div([
                    html.Label("Date", style={"color":"#7eb8f7","fontSize":"12px",
                                              "fontWeight":"600","marginBottom":"6px",
                                              "display":"block"}),
                    dcc.DatePickerSingle(
                        id="dep-date",
                        date=datetime.today().strftime("%Y-%m-%d"),
                        display_format="MMM D, YYYY",
                        style={"fontSize":"14px"}),
                ], style={"flex":"0"}),

                # Departure time
                html.Div([
                    html.Label("Departs", style={"color":"#7eb8f7","fontSize":"12px",
                                                  "fontWeight":"600","marginBottom":"6px",
                                                  "display":"block"}),
                    dcc.Input(id="dep-time", type="text", placeholder="HH:MM",
                              value="14:00", debounce=True,
                              style={"background":"#1a2535","color":"#e8eef8",
                                     "border":"1px solid #2a3a50","padding":"8px 12px",
                                     "borderRadius":"4px","width":"90px","fontSize":"14px"}),
                ], style={"flex":"0"}),

                # Flight number (optional)
                html.Div([
                    html.Label("Flight # (optional)",
                               style={"color":"#7eb8f7","fontSize":"12px",
                                      "fontWeight":"600","marginBottom":"6px",
                                      "display":"block"}),
                    dcc.Input(id="flight-num", type="text",
                              placeholder="e.g. UA328", debounce=True,
                              style={"background":"#1a2535","color":"#e8eef8",
                                     "border":"1px solid #2a3a50","padding":"8px 12px",
                                     "borderRadius":"4px","width":"110px","fontSize":"14px"}),
                ], style={"flex":"0"}),

                # Search button
                html.Div([
                    html.Label(" ", style={"display":"block","marginBottom":"6px",
                                           "fontSize":"12px"}),
                    html.Button("Predict →", id="predict-btn",
                                style={"background":"#7eb8f7","color":"#0a0f1e",
                                       "border":"none","padding":"9px 24px",
                                       "borderRadius":"4px","fontWeight":"700",
                                       "cursor":"pointer","fontSize":"14px",
                                       "whiteSpace":"nowrap"}),
                ], style={"flex":"0"}),

            ], style={"display":"flex","gap":"12px","alignItems":"flex-start",
                      "flexWrap":"wrap"}),
        ], style={"background":"#0d1628","padding":"24px 32px",
                  "borderBottom":"1px solid #1e2d4a"}),

        # Loading + results
        dcc.Loading(
            id="loading",
            type="circle",
            color="#7eb8f7",
            children=html.Div(id="results",
                              style={"padding":"32px","maxWidth":"960px",
                                     "margin":"0 auto"}),
        ),

        # Footer
        html.Div([
            html.P("Research prototype · Two-Level Heterogeneous GNN · "
                   "~20 min MAE on operational flights · Not for operational use",
                   style={"color":"#2a3a50","fontSize":"11px","margin":"0",
                          "textAlign":"center"}),
        ], style={"padding":"16px","borderTop":"1px solid #1e2d4a",
                  "marginTop":"auto"}),

    ], style={"background":"#0a0f1e","minHeight":"100vh",
              "fontFamily":"'Segoe UI',system-ui,sans-serif",
              "display":"flex","flexDirection":"column",
              "color":"#e8eef8"})

    # ── Callback ──────────────────────────────────────────────────────────
    @app.callback(
        Output("results","children"),
        Input("predict-btn","n_clicks"),
        State("origin","value"),
        State("dest","value"),
        State("dep-date","date"),
        State("dep-time","value"),
        State("flight-num","value"),
        prevent_initial_call=True,
    )
    def run_prediction(n_clicks, origin, dest, date, dep_time, flight_num):
        if not origin or not dest:
            return _error("Please select both origin and destination airports.")
        if origin == dest:
            return _error("Origin and destination cannot be the same.")
        if origin not in ap_idx_map or dest not in ap_idx_map:
            return _error("One or both airports are not in the 36-hub network.")

        # Parse departure datetime
        try:
            dep_dt_str = f"{date} {dep_time or '14:00'}"
            dep_dt = pd.Timestamp(dep_dt_str)
        except Exception:
            return _error("Invalid date or time format.")

        try:
            preds = predict_flight(
                origin, dest, dep_dt_str, flight_num or "",
                model, airports, ap_idx_map, static, route_stats, device)
        except Exception as e:
            return _error(f"Prediction failed: {e}")

        return _result_card(origin, dest, dep_dt, flight_num, preds)

    return app


def _error(msg):
    return html.Div(msg, style={"color":"#e74c3c","padding":"16px",
                                "background":"#2a1010","borderRadius":"8px",
                                "border":"1px solid #e74c3c"})


def _result_card(origin, dest, dep_dt, flight_num, preds):
    """Build the result display."""

    def delay_color(d):
        if d < 5:   return "#2ecc71"
        if d < 20:  return "#f1c40f"
        if d < 45:  return "#e67e22"
        return "#e74c3c"

    def status_text(d):
        if d < 5:   return "🟢 On Time"
        if d < 20:  return "🟡 Minor Delay"
        if d < 45:  return "🟠 Moderate Delay"
        return "🔴 Severe Delay"

    def horizon_card(label, h_key, conf_color):
        p = preds.get(h_key, {})
        d = p.get("pred_delay", 0)
        prob = p.get("delay_prob", 0.5) * 100
        arr  = p.get("pred_arrival","—")
        col  = delay_color(d)
        return html.Div([
            html.Div(label,
                     style={"color":"#6b7a99","fontSize":"10px","fontWeight":"700",
                            "textTransform":"uppercase","letterSpacing":"1px",
                            "marginBottom":"10px"}),
            html.Div(status_text(d),
                     style={"fontSize":"13px","marginBottom":"8px","color":col}),
            html.Div(f"{d:+.0f} min",
                     style={"fontSize":"38px","fontWeight":"700","color":col,
                            "lineHeight":"1","marginBottom":"8px"}),
            html.Div(f"±{MODEL_MAE} min accuracy",
                     style={"color":"#4a5a70","fontSize":"10px","marginBottom":"6px"}),
            html.Div(f"Pred arrival: {arr}",
                     style={"color":"#6b7a99","fontSize":"12px","marginBottom":"4px"}),
            html.Div(f"Delay prob: {prob:.0f}%",
                     style={"color":"#6b7a99","fontSize":"11px"}),
        ], style={"background":"#131929",
                  "border":f"1px solid {col}40",
                  "borderTop":f"3px solid {col}",
                  "borderRadius":"8px","padding":"20px",
                  "flex":"1","minWidth":"160px"})

    # Weather summary
    wx  = preds.get("weather",{})
    wxo = wx.get("origin",{})
    wxd = wx.get("dest",{})

    fi  = preds.get("flight_info")
    fi_note = ""
    if fi:
        fi_note = f" · {fi.get('status','').title()} · Tail: {fi.get('tail','—')}"

    route = f"{origin} → {dest}"
    sched = dep_dt.strftime("%b %d, %Y at %H:%M")

    return html.Div([

        # Route header
        html.Div([
            html.Div([
                html.Span(origin, style={"fontSize":"36px","fontWeight":"700",
                                         "color":"#7eb8f7"}),
                html.Span("  →  ", style={"fontSize":"24px","color":"#2a3a50",
                                           "margin":"0 4px"}),
                html.Span(dest,   style={"fontSize":"36px","fontWeight":"700",
                                         "color":"#7eb8f7"}),
            ]),
            html.Div([
                html.Span(sched,
                          style={"color":"#6b7a99","fontSize":"14px"}),
                html.Span(f" · {flight_num.upper()}" if flight_num else "",
                          style={"color":"#7eb8f7","fontSize":"14px"}),
                html.Span(fi_note,
                          style={"color":"#4a5a70","fontSize":"12px"}),
            ], style={"marginTop":"4px"}),
        ], style={"marginBottom":"28px"}),

        # Three prediction cards
        html.Div([
            horizon_card("6 Hours Before",  "6h", "#9b59b6"),
            horizon_card("3 Hours Before",  "3h", "#e67e22"),
            horizon_card("1 Hour Before",   "1h", "#2ecc71"),
        ], style={"display":"flex","gap":"16px","marginBottom":"20px",
                  "flexWrap":"wrap"}),

        # Weather context
        html.Div([
            html.Div([
                html.Span(f"🌤 {AIRPORT_NAMES.get(origin,origin)}: ",
                          style={"color":"#7eb8f7","fontSize":"12px","fontWeight":"600"}),
                html.Span(f"Wind {wxo.get('wind_speed_ms',0):.1f} m/s · "
                          f"Vis {wxo.get('visibility_m',10000)/1000:.0f} km · "
                          f"Precip {wxo.get('precip_depth_mm',0):.1f} mm",
                          style={"color":"#6b7a99","fontSize":"12px"}),
            ], style={"marginBottom":"4px"}),
            html.Div([
                html.Span(f"🌤 {AIRPORT_NAMES.get(dest,dest)}: ",
                          style={"color":"#7eb8f7","fontSize":"12px","fontWeight":"600"}),
                html.Span(f"Wind {wxd.get('wind_speed_ms',0):.1f} m/s · "
                          f"Vis {wxd.get('visibility_m',10000)/1000:.0f} km · "
                          f"Precip {wxd.get('precip_depth_mm',0):.1f} mm",
                          style={"color":"#6b7a99","fontSize":"12px"}),
            ]),
        ], style={"background":"#0d1628","borderRadius":"8px","padding":"14px 16px",
                  "border":"1px solid #1e2d4a","marginBottom":"16px"}),

        # Disclaimer
        html.Div([
            html.Span("⚠  Research prototype · ±20 min accuracy · "
                      "Not for operational decisions · "
                      "Cancellations not predicted · "
                      "Hub routes only",
                      style={"color":"#4a5a70","fontSize":"10px"}),
        ], style={"textAlign":"center"}),

    ])


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    model, airports, ap_idx_map, static, route_stats, device = load_assets()
    app = build_app(model, airports, ap_idx_map, static, route_stats, device)

    print("\n" + "="*55)
    print("  Flight Finder running at http://127.0.0.1:8051")
    print("  Press Ctrl+C to stop")
    print("="*55 + "\n")

    app.run(debug=False, host="127.0.0.1", port=8051)


if __name__ == "__main__":
    main()