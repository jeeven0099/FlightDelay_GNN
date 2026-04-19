"""
STEP 11 - VALIDATION DAY REPLAY DASHBOARD
=========================================
Interactive website for replaying one validation day from the final
departure-delay model. This is designed for portfolio/demo use:

- timeline playback across a full day of hourly snapshots
- US map colored by predicted severity or correctness
- full sortable/filterable table with flight metadata
- explicit correctness fields for both tier matching and severe alerts

The dashboard uses the saved validation evaluation CSV rather than rerunning
the model live. On first run for a given date, it extracts and caches a small
one-day parquet under outputs/ for fast subsequent startup.

Usage:
  python 11_val_day_replay_dashboard.py
  python 11_val_day_replay_dashboard.py --date 2021-11-28
  python 11_val_day_replay_dashboard.py --port 8062 --threshold 0.60
"""

from __future__ import annotations

import os

# Avoid duplicate OpenMP init issues when pandas + torch are both loaded.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch

try:
    import dash
    from dash import Dash, Input, Output, State, dcc, html, dash_table
    import dash_bootstrap_components as dbc
    DASH_IMPORT_ERROR = None
except ImportError as e:
    dash = None
    Dash = None
    Input = Output = State = None
    dcc = html = dash_table = None
    dbc = None
    DASH_IMPORT_ERROR = e


BASE_DIR = r"C:\Users\user\Desktop\Airline_Graphs_Project"
GRAPH_DATA_DIR = os.path.join(BASE_DIR, "graph_data")
EVAL_DIR = os.path.join(BASE_DIR, "evaluation")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CACHE_DIR = os.path.join(EVAL_DIR, "demo_cache")

DEFAULT_VAL_CSV = os.path.join(
    EVAL_DIR, "eval_best_model_dep_12k_ordinal_ep25_val.csv"
)
DEFAULT_DATE = "2021-11-28"
DEFAULT_THRESHOLD = 0.60
DEFAULT_PORT = 8062

AIRPORT_COORDS = {
    "ANC": (61.174, -149.996), "ATL": (33.640, -84.427),
    "BNA": (36.124, -86.678), "BOS": (42.365, -71.009),
    "BWI": (39.175, -76.668), "CLE": (41.411, -81.849),
    "CLT": (35.214, -80.943), "CMH": (39.998, -82.892),
    "DEN": (39.856, -104.674), "DFW": (32.897, -97.038),
    "DTW": (42.212, -83.353), "EWR": (40.692, -74.174),
    "FLL": (26.072, -80.150), "HOU": (29.645, -95.279),
    "IAD": (38.944, -77.456), "IAH": (29.984, -95.341),
    "IND": (39.717, -86.294), "JFK": (40.639, -73.779),
    "LAS": (36.080, -115.152), "LAX": (33.943, -118.408),
    "LGA": (40.777, -73.873), "MCI": (39.298, -94.714),
    "MCO": (28.429, -81.309), "MIA": (25.796, -80.287),
    "MKE": (42.947, -87.897), "MSP": (44.882, -93.222),
    "ORD": (41.978, -87.905), "PHL": (39.872, -75.241),
    "PHX": (33.437, -112.008), "PIT": (40.492, -80.233),
    "SAN": (32.734, -117.190), "SEA": (47.449, -122.309),
    "SFO": (37.619, -122.375), "SJC": (37.363, -121.929),
    "SLC": (40.788, -111.978), "TPA": (27.975, -82.533),
}

TIER_BINS = [-1e9, 0, 15, 60, 120, 240, 720, 1e9]
TIER_LABELS = [
    "Early / On Time",
    "Minor (0-15)",
    "Moderate (15-60)",
    "Heavy (60-120)",
    "Severe (120-240)",
    "Extreme (240-720)",
    "Ultra (720+)",
]
TIER_COLORS = {
    "Early / On Time": "#2D6A4F",
    "Minor (0-15)": "#74A57F",
    "Moderate (15-60)": "#D9A441",
    "Heavy (60-120)": "#E76F51",
    "Severe (120-240)": "#C44536",
    "Extreme (240-720)": "#8D2B3A",
    "Ultra (720+)": "#4A0D1A",
}
CORRECTNESS_COLORS = {
    "Exact Tier Match": "#2D6A4F",
    "Tier Miss": "#C44536",
}
ALERT_RESULT_COLORS = {
    "TP": "#2D6A4F",
    "FP": "#C44536",
    "FN": "#B08D57",
    "TN": "#8FA8A3",
}
HORIZON_LABELS = {
    0: "<1h",
    1: "1-3h",
    3: "3-6h",
    6: ">6h",
}
SEVERITY_FILTERS = {
    "all": -1,
    "heavy_plus": 3,
    "severe_plus": 4,
    "extreme_plus": 5,
    "ultra_only": 6,
}

PAPER_BG = "#F7F2E8"
CARD_BG = "#FFFDF8"
INK = "#16202A"
SUBTLE = "#5B6770"
GRID = "#D8CFC3"
ACCENT = "#1F4E5F"


def cache_path_for(date_str: str, split: str) -> str:
    return os.path.join(CACHE_DIR, f"demo_replay_{split}_{date_str}.parquet")


def snapshot_time_cache_path(split: str) -> str:
    return os.path.join(CACHE_DIR, f"snapshot_times_{split}.parquet")


def bucket_delay(series: pd.Series) -> pd.Series:
    return pd.cut(series, bins=TIER_BINS, labels=TIER_LABELS, right=False).astype(str)


def load_snapshot_time_map(split: str) -> pd.DataFrame:
    cache_path = snapshot_time_cache_path(split)
    if os.path.exists(cache_path):
        snap_map = pd.read_parquet(cache_path)
        snap_map["snapshot_time"] = pd.to_datetime(snap_map["snapshot_time"])
        return snap_map

    snap_file = os.path.join(GRAPH_DATA_DIR, f"snapshots_{split}.pt")
    snapshots = torch.load(snap_file, map_location="cpu", weights_only=False)
    snap_map = pd.DataFrame(
        {
            "snap_idx": np.arange(len(snapshots), dtype=np.int32),
            "snapshot_time": [
                pd.Timestamp(s["airport"].snapshot_time) for s in snapshots
            ],
        }
    )
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        snap_map.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"  warning: could not cache snapshot times ({e})")
    return snap_map


def build_day_cache(
    date_str: str,
    eval_csv: str,
    split: str,
    severe_threshold: float,
    force_rebuild: bool = False,
) -> str:
    out_path = cache_path_for(date_str, split)
    if os.path.exists(out_path) and not force_rebuild:
        return out_path

    print(f"Preparing replay cache for {date_str} from {os.path.basename(eval_csv)} ...")

    flight_lookup = pd.read_parquet(
        os.path.join(GRAPH_DATA_DIR, "flight_lookup.parquet"),
        columns=[
            "flight_id",
            "ORIGIN",
            "DEST",
            "dep_datetime",
            "arr_datetime",
            "Tail_Number",
            "Operating_Airline",
        ],
    )
    flight_lookup["dep_datetime"] = pd.to_datetime(flight_lookup["dep_datetime"])
    flight_lookup["arr_datetime"] = pd.to_datetime(flight_lookup["arr_datetime"])

    target_date = pd.Timestamp(date_str).date()
    day_lookup = flight_lookup[flight_lookup["dep_datetime"].dt.date == target_date].copy()
    if len(day_lookup) == 0:
        raise ValueError(f"No flights found in flight_lookup for {date_str}")

    day_ids = set(day_lookup["flight_id"].astype(np.int64).tolist())
    usecols = ["flight_id", "snap_idx", "horizon_h", "pred", "severe_prob", "actual", "abs_err", "sq_err"]

    chunks = []
    for i, chunk in enumerate(pd.read_csv(eval_csv, usecols=usecols, chunksize=500_000)):
        keep = chunk["flight_id"].isin(day_ids)
        if keep.any():
            chunks.append(chunk.loc[keep].copy())
        if (i + 1) % 4 == 0:
            print(f"  scanned {((i + 1) * 500_000):,}+ rows ...")

    if not chunks:
        raise ValueError(f"No evaluation rows found for {date_str} in {eval_csv}")

    df = pd.concat(chunks, ignore_index=True)
    snap_map = load_snapshot_time_map(split)

    df = df.merge(day_lookup, on="flight_id", how="left")
    df = df.merge(snap_map, on="snap_idx", how="left")
    df["snapshot_time"] = pd.to_datetime(df["snapshot_time"])
    df["dep_datetime"] = pd.to_datetime(df["dep_datetime"])
    df["arr_datetime"] = pd.to_datetime(df["arr_datetime"])
    df = df.dropna(subset=["snapshot_time", "dep_datetime"]).copy()

    df["hours_to_departure"] = (
        (df["dep_datetime"] - df["snapshot_time"]).dt.total_seconds() / 3600.0
    )
    df["horizon_label"] = df["horizon_h"].map(HORIZON_LABELS)
    df["pred_tier"] = bucket_delay(df["pred"])
    df["actual_tier"] = bucket_delay(df["actual"])
    df["pred_tier_order"] = df["pred_tier"].map({label: i for i, label in enumerate(TIER_LABELS)})
    df["actual_tier_order"] = df["actual_tier"].map({label: i for i, label in enumerate(TIER_LABELS)})
    df["tier_match"] = df["pred_tier"] == df["actual_tier"]
    df["correctness"] = np.where(df["tier_match"], "Exact Tier Match", "Tier Miss")
    df["severe_alert"] = df["severe_prob"] >= severe_threshold
    df["actual_severe"] = df["actual"] >= 120.0
    df["alert_result"] = np.select(
        [
            df["severe_alert"] & df["actual_severe"],
            df["severe_alert"] & ~df["actual_severe"],
            ~df["severe_alert"] & df["actual_severe"],
        ],
        ["TP", "FP", "FN"],
        default="TN",
    )

    df["route"] = df["ORIGIN"] + " -> " + df["DEST"]
    df["snapshot_label"] = df["snapshot_time"].dt.strftime("%Y-%m-%d %H:%M")
    df["dep_time_label"] = df["dep_datetime"].dt.strftime("%H:%M")
    df["arr_time_label"] = df["arr_datetime"].dt.strftime("%H:%M")

    df["origin_lat"] = df["ORIGIN"].map(lambda x: AIRPORT_COORDS.get(x, (np.nan, np.nan))[0])
    df["origin_lon"] = df["ORIGIN"].map(lambda x: AIRPORT_COORDS.get(x, (np.nan, np.nan))[1])
    df["dest_lat"] = df["DEST"].map(lambda x: AIRPORT_COORDS.get(x, (np.nan, np.nan))[0])
    df["dest_lon"] = df["DEST"].map(lambda x: AIRPORT_COORDS.get(x, (np.nan, np.nan))[1])
    df = df.dropna(subset=["origin_lat", "origin_lon", "dest_lat", "dest_lon"]).copy()

    keep_cols = [
        "flight_id",
        "snap_idx",
        "snapshot_time",
        "snapshot_label",
        "horizon_h",
        "horizon_label",
        "hours_to_departure",
        "ORIGIN",
        "DEST",
        "route",
        "dep_datetime",
        "dep_time_label",
        "arr_datetime",
        "arr_time_label",
        "Tail_Number",
        "Operating_Airline",
        "pred",
        "pred_tier",
        "pred_tier_order",
        "severe_prob",
        "severe_alert",
        "actual",
        "actual_tier",
        "actual_tier_order",
        "actual_severe",
        "abs_err",
        "sq_err",
        "tier_match",
        "correctness",
        "alert_result",
        "origin_lat",
        "origin_lon",
        "dest_lat",
        "dest_lon",
    ]
    df = df[keep_cols].sort_values(["snapshot_time", "pred_tier_order", "severe_prob"], ascending=[True, False, False])

    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"  saved {len(df):,} replay rows -> {out_path}")
    return out_path


def make_kpi_card(title: str, value_id: str, subtitle_id: str = None) -> dbc.Card:
    body = [
        html.Div(title, style={"fontSize": "0.85rem", "letterSpacing": "0.04em", "textTransform": "uppercase", "color": SUBTLE}),
        html.Div(id=value_id, style={"fontSize": "2rem", "fontWeight": "700", "color": INK}),
    ]
    if subtitle_id is not None:
        body.append(html.Div(id=subtitle_id, style={"fontSize": "0.9rem", "color": SUBTLE}))
    return dbc.Card(
        dbc.CardBody(body, style={"padding": "14px 16px"}),
        style={"background": CARD_BG, "border": f"1px solid {GRID}", "borderRadius": "18px", "boxShadow": "0 8px 24px rgba(0,0,0,0.04)"},
    )


def build_map_figure(df: pd.DataFrame, color_mode: str, map_limit: str) -> go.Figure:
    plot_df = df.copy()
    if map_limit != "all":
        plot_df = plot_df.sort_values(
            ["pred_tier_order", "severe_prob", "abs_err"],
            ascending=[False, False, False],
        ).head(int(map_limit))

    if color_mode == "correctness":
        color_col = "correctness"
        color_map = CORRECTNESS_COLORS
        groups = ["Exact Tier Match", "Tier Miss"]
    elif color_mode == "alert":
        color_col = "alert_result"
        color_map = ALERT_RESULT_COLORS
        groups = ["TP", "FP", "FN", "TN"]
    else:
        color_col = "pred_tier"
        color_map = TIER_COLORS
        groups = TIER_LABELS

    fig = go.Figure()

    for group in groups:
        sub = plot_df[plot_df[color_col] == group]
        if len(sub) == 0:
            continue

        lon_lines: List[float] = []
        lat_lines: List[float] = []
        for row in sub.itertuples(index=False):
            lon_lines.extend([row.origin_lon, row.dest_lon, None])
            lat_lines.extend([row.origin_lat, row.dest_lat, None])

        fig.add_trace(
            go.Scattergeo(
                lon=lon_lines,
                lat=lat_lines,
                mode="lines",
                line={"width": 1.2, "color": color_map[group]},
                opacity=0.38,
                name=group,
                hoverinfo="skip",
                showlegend=True,
            )
        )

        hover = (
            "Flight " + sub["flight_id"].astype(str)
            + " · " + sub["Operating_Airline"].fillna("")
            + "<br>" + sub["route"]
            + "<br>Dep " + sub["dep_time_label"]
            + " · Horizon " + sub["horizon_label"].astype(str)
            + "<br>Pred " + sub["pred"].round(1).astype(str)
            + " min · " + sub["pred_tier"]
            + "<br>Actual " + sub["actual"].round(1).astype(str)
            + " min · " + sub["actual_tier"]
            + "<br>Severe p=" + sub["severe_prob"].round(3).astype(str)
            + "<br>Alert " + sub["alert_result"]
        )

        fig.add_trace(
            go.Scattergeo(
                lon=sub["dest_lon"],
                lat=sub["dest_lat"],
                mode="markers",
                marker={
                    "size": np.clip(6 + sub["severe_prob"].to_numpy() * 10, 6, 14),
                    "color": color_map[group],
                    "opacity": 0.85,
                    "line": {"width": 0.4, "color": "#ffffff"},
                },
                text=hover,
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
                name=group,
            )
        )

    airport_codes = sorted(set(plot_df["ORIGIN"]).union(set(plot_df["DEST"])))
    ap_lat = [AIRPORT_COORDS[ap][0] for ap in airport_codes]
    ap_lon = [AIRPORT_COORDS[ap][1] for ap in airport_codes]
    fig.add_trace(
        go.Scattergeo(
            lon=ap_lon,
            lat=ap_lat,
            text=airport_codes,
            mode="markers+text",
            marker={"size": 4, "color": ACCENT, "opacity": 0.75},
            textfont={"size": 9, "color": INK},
            textposition="top center",
            hovertemplate="%{text}<extra></extra>",
            name="Airports",
            showlegend=False,
        )
    )

    fig.update_layout(
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PAPER_BG,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "left",
            "x": 0.01,
            "font": {"size": 11},
        },
        geo={
            "scope": "usa",
            "projection_type": "albers usa",
            "showland": True,
            "landcolor": "#F2ECE2",
            "showlakes": True,
            "lakecolor": "#DDEBF3",
            "showocean": True,
            "oceancolor": "#E8F1F6",
            "subunitcolor": "#D2C7B8",
            "countrycolor": "#D2C7B8",
            "bgcolor": PAPER_BG,
        },
    )
    return fig


def build_layout(unique_snapshots: List[pd.Timestamp], date_str: str, severe_threshold: float) -> html.Div:
    slider_marks = {
        i: ts.strftime("%H:%M") if i % 3 == 0 or i == len(unique_snapshots) - 1 else ""
        for i, ts in enumerate(unique_snapshots)
    }

    return dbc.Container(
        fluid=True,
        style={"background": PAPER_BG, "minHeight": "100vh", "padding": "22px 26px 28px"},
        children=[
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                "Flight Delay Replay",
                                style={"fontSize": "2.1rem", "fontWeight": "800", "color": INK, "fontFamily": "Georgia"},
                            ),
                            html.Div(
                                f"Validation-day replay for {date_str} using the ordinal departure-delay model · severe alert threshold = {severe_threshold:.2f} · timeline includes prior-evening snapshots needed for early departures",
                                style={"fontSize": "1rem", "color": SUBTLE, "marginTop": "4px"},
                            ),
                        ],
                        md=8,
                    ),
                    dbc.Col(
                        html.Div(
                            "Historical replay, not live inference",
                            style={
                                "textAlign": "right",
                                "fontSize": "0.95rem",
                                "fontWeight": "600",
                                "color": ACCENT,
                                "paddingTop": "10px",
                            },
                        ),
                        md=4,
                    ),
                ],
                style={"marginBottom": "14px"},
            ),
            dbc.Row(
                [
                    dbc.Col(make_kpi_card("Snapshot", "kpi-snapshot", "kpi-snapshot-sub"), md=3),
                    dbc.Col(make_kpi_card("Flights Visible", "kpi-count", "kpi-count-sub"), md=3),
                    dbc.Col(make_kpi_card("Severe Alerts", "kpi-alerts", "kpi-alerts-sub"), md=3),
                    dbc.Col(make_kpi_card("Tier Accuracy", "kpi-accuracy", "kpi-accuracy-sub"), md=3),
                ],
                style={"marginBottom": "14px"},
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                dbc.Button(
                                                    "Play",
                                                    id="play-pause-btn",
                                                    color="dark",
                                                    n_clicks=0,
                                                    style={"width": "100%"},
                                                ),
                                                md=2,
                                            ),
                                            dbc.Col(
                                                dcc.Dropdown(
                                                    id="color-mode",
                                                    options=[
                                                        {"label": "Color by predicted severity", "value": "tier"},
                                                        {"label": "Color by exact tier correctness", "value": "correctness"},
                                                        {"label": "Color by severe alert result", "value": "alert"},
                                                    ],
                                                    value="tier",
                                                    clearable=False,
                                                ),
                                                md=4,
                                            ),
                                            dbc.Col(
                                                dcc.Dropdown(
                                                    id="severity-filter",
                                                    options=[
                                                        {"label": "Show all", "value": "all"},
                                                        {"label": "Heavy+ only", "value": "heavy_plus"},
                                                        {"label": "Severe+ only", "value": "severe_plus"},
                                                        {"label": "Extreme+ only", "value": "extreme_plus"},
                                                        {"label": "Ultra only", "value": "ultra_only"},
                                                    ],
                                                    value="all",
                                                    clearable=False,
                                                ),
                                                md=3,
                                            ),
                                            dbc.Col(
                                                dcc.Dropdown(
                                                    id="map-limit",
                                                    options=[
                                                        {"label": "Top 200 routes on map", "value": "200"},
                                                        {"label": "Top 500 routes on map", "value": "500"},
                                                        {"label": "All visible routes on map", "value": "all"},
                                                    ],
                                                    value="500",
                                                    clearable=False,
                                                ),
                                                md=3,
                                            ),
                                        ],
                                        style={"marginBottom": "12px"},
                                    ),
                                    dcc.Slider(
                                        id="snapshot-slider",
                                        min=0,
                                        max=len(unique_snapshots) - 1,
                                        step=1,
                                        value=0,
                                        marks=slider_marks,
                                    ),
                                ]
                            ),
                            style={"background": CARD_BG, "border": f"1px solid {GRID}", "borderRadius": "18px", "boxShadow": "0 8px 24px rgba(0,0,0,0.04)"},
                        ),
                        md=12,
                        style={"marginBottom": "14px"},
                    )
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                dcc.Tabs(
                                    value="tab-map",
                                    colors={"border": GRID, "primary": ACCENT, "background": CARD_BG},
                                    children=[
                                        dcc.Tab(
                                            label="Map Replay",
                                            value="tab-map",
                                            style={"backgroundColor": CARD_BG, "color": SUBTLE, "padding": "10px 14px", "borderBottom": f"1px solid {GRID}"},
                                            selected_style={"backgroundColor": "#EFE5D7", "color": INK, "fontWeight": "700", "padding": "10px 14px", "borderTop": f"3px solid {ACCENT}"},
                                            children=[
                                                html.Div(
                                                    dcc.Graph(
                                                        id="replay-map",
                                                        config={"displayModeBar": False},
                                                        style={"height": "560px"},
                                                    ),
                                                    style={"paddingTop": "10px"},
                                                )
                                            ],
                                        ),
                                        dcc.Tab(
                                            label="Flight Table",
                                            value="tab-table",
                                            style={"backgroundColor": CARD_BG, "color": SUBTLE, "padding": "10px 14px", "borderBottom": f"1px solid {GRID}"},
                                            selected_style={"backgroundColor": "#EFE5D7", "color": INK, "fontWeight": "700", "padding": "10px 14px", "borderTop": f"3px solid {ACCENT}"},
                                            children=[
                                                html.Div(
                                                    [
                                                        html.Div(
                                                            "Flight Table",
                                                            style={"fontSize": "1.2rem", "fontWeight": "700", "color": INK, "marginBottom": "10px", "marginTop": "10px"},
                                                        ),
                                                        dash_table.DataTable(
                                                            id="flight-table",
                                                            columns=[],
                                                            data=[],
                                                            sort_action="native",
                                                            filter_action="native",
                                                            page_action="native",
                                                            page_size=16,
                                                            style_table={"overflowX": "auto"},
                                                            style_header={
                                                                "backgroundColor": "#EFE5D7",
                                                                "color": INK,
                                                                "fontWeight": "700",
                                                                "border": f"1px solid {GRID}",
                                                            },
                                                            style_cell={
                                                                "backgroundColor": CARD_BG,
                                                                "color": INK,
                                                                "border": f"1px solid {GRID}",
                                                                "fontSize": "0.88rem",
                                                                "padding": "6px 8px",
                                                                "textAlign": "left",
                                                                "minWidth": "90px",
                                                                "maxWidth": "220px",
                                                                "whiteSpace": "normal",
                                                            },
                                                            style_data_conditional=[
                                                                {
                                                                    "if": {"filter_query": "{correctness} = 'Exact Tier Match'", "column_id": "correctness"},
                                                                    "backgroundColor": "#E5F2EA",
                                                                    "color": "#1E5132",
                                                                    "fontWeight": "700",
                                                                },
                                                                {
                                                                    "if": {"filter_query": "{correctness} = 'Tier Miss'", "column_id": "correctness"},
                                                                    "backgroundColor": "#FBE6E1",
                                                                    "color": "#8D2B3A",
                                                                    "fontWeight": "700",
                                                                },
                                                                {
                                                                    "if": {"filter_query": "{alert_result} = 'TP'", "column_id": "alert_result"},
                                                                    "backgroundColor": "#E5F2EA",
                                                                    "color": "#1E5132",
                                                                    "fontWeight": "700",
                                                                },
                                                                {
                                                                    "if": {"filter_query": "{alert_result} = 'FP'", "column_id": "alert_result"},
                                                                    "backgroundColor": "#FBE6E1",
                                                                    "color": "#8D2B3A",
                                                                    "fontWeight": "700",
                                                                },
                                                                {
                                                                    "if": {"filter_query": "{alert_result} = 'FN'", "column_id": "alert_result"},
                                                                    "backgroundColor": "#F7EDD9",
                                                                    "color": "#8A6C3D",
                                                                    "fontWeight": "700",
                                                                },
                                                            ],
                                                        ),
                                                    ]
                                                )
                                            ],
                                        ),
                                    ],
                                )
                            ),
                            style={"background": CARD_BG, "border": f"1px solid {GRID}", "borderRadius": "18px", "boxShadow": "0 8px 24px rgba(0,0,0,0.04)"},
                        ),
                        md=12,
                    )
                ]
            ),
            dcc.Interval(id="play-interval", interval=1200, n_intervals=0, disabled=True),
            dcc.Store(id="is-playing", data=False),
        ],
    )


def main():
    if DASH_IMPORT_ERROR is not None:
        raise SystemExit(
            "This dashboard needs Dash. Install with: "
            "pip install dash dash-bootstrap-components plotly"
        )

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=DEFAULT_DATE, help="Validation date to replay, e.g. 2021-11-28")
    parser.add_argument("--csv", default=DEFAULT_VAL_CSV, help="Validation evaluation CSV to replay")
    parser.add_argument("--split", default="val", choices=["val", "test"], help="Snapshot split backing the replay")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Severe alert threshold for severe_prob")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Dash port")
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild the cached day parquet")
    args = parser.parse_args()

    cache_path = build_day_cache(
        date_str=args.date,
        eval_csv=args.csv,
        split=args.split,
        severe_threshold=args.threshold,
        force_rebuild=args.force_rebuild,
    )
    day_df = pd.read_parquet(cache_path)
    day_df["snapshot_time"] = pd.to_datetime(day_df["snapshot_time"])
    day_df["dep_datetime"] = pd.to_datetime(day_df["dep_datetime"])
    day_df["arr_datetime"] = pd.to_datetime(day_df["arr_datetime"])

    unique_snapshots = sorted(day_df["snapshot_time"].dropna().unique().tolist())
    snapshot_lookup = {i: ts for i, ts in enumerate(unique_snapshots)}

    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.title = "Flight Delay Replay"
    app.layout = build_layout(unique_snapshots, args.date, args.threshold)

    @app.callback(
        Output("is-playing", "data"),
        Output("play-interval", "disabled"),
        Output("play-pause-btn", "children"),
        Input("play-pause-btn", "n_clicks"),
        State("is-playing", "data"),
        prevent_initial_call=True,
    )
    def toggle_play(n_clicks, is_playing):
        next_state = not is_playing
        return next_state, (not next_state), ("Pause" if next_state else "Play")

    @app.callback(
        Output("snapshot-slider", "value"),
        Input("play-interval", "n_intervals"),
        State("snapshot-slider", "value"),
        State("is-playing", "data"),
    )
    def advance_slider(n_intervals, slider_value, is_playing):
        if not is_playing:
            return slider_value
        if slider_value >= len(unique_snapshots) - 1:
            return 0
        return slider_value + 1

    @app.callback(
        Output("kpi-snapshot", "children"),
        Output("kpi-snapshot-sub", "children"),
        Output("kpi-count", "children"),
        Output("kpi-count-sub", "children"),
        Output("kpi-alerts", "children"),
        Output("kpi-alerts-sub", "children"),
        Output("kpi-accuracy", "children"),
        Output("kpi-accuracy-sub", "children"),
        Output("replay-map", "figure"),
        Output("flight-table", "columns"),
        Output("flight-table", "data"),
        Input("snapshot-slider", "value"),
        Input("color-mode", "value"),
        Input("severity-filter", "value"),
        Input("map-limit", "value"),
    )
    def update_view(snapshot_pos, color_mode, severity_filter, map_limit):
        snapshot_time = pd.Timestamp(snapshot_lookup[snapshot_pos])
        snap_df = day_df[day_df["snapshot_time"] == snapshot_time].copy()

        severity_cut = SEVERITY_FILTERS[severity_filter]
        if severity_cut >= 0:
            snap_df = snap_df[snap_df["pred_tier_order"] >= severity_cut].copy()

        flights_visible = len(snap_df)
        severe_alerts = int(snap_df["severe_alert"].sum()) if flights_visible else 0
        tier_acc = float(snap_df["tier_match"].mean()) if flights_visible else 0.0
        avg_err = float(snap_df["abs_err"].mean()) if flights_visible else 0.0
        avg_pred = float(snap_df["pred"].mean()) if flights_visible else 0.0

        alert_tp = int(((snap_df["alert_result"] == "TP")).sum()) if flights_visible else 0
        alert_fp = int(((snap_df["alert_result"] == "FP")).sum()) if flights_visible else 0
        alert_precision = alert_tp / (alert_tp + alert_fp) if (alert_tp + alert_fp) else 0.0

        fig = build_map_figure(snap_df, color_mode=color_mode, map_limit=map_limit)

        table_df = snap_df.sort_values(
            ["pred_tier_order", "severe_prob", "abs_err"],
            ascending=[False, False, False],
        ).copy()
        table_df["pred"] = table_df["pred"].round(1)
        table_df["actual"] = table_df["actual"].round(1)
        table_df["abs_err"] = table_df["abs_err"].round(1)
        table_df["severe_prob"] = table_df["severe_prob"].round(3)
        table_df["hours_to_departure"] = table_df["hours_to_departure"].round(2)

        display_cols = [
            "flight_id",
            "Operating_Airline",
            "Tail_Number",
            "ORIGIN",
            "DEST",
            "route",
            "snapshot_label",
            "dep_time_label",
            "arr_time_label",
            "horizon_label",
            "hours_to_departure",
            "pred",
            "pred_tier",
            "severe_prob",
            "severe_alert",
            "actual",
            "actual_tier",
            "correctness",
            "alert_result",
            "abs_err",
        ]

        columns = [{"name": c, "id": c} for c in display_cols]
        data = table_df[display_cols].to_dict("records")

        return (
            snapshot_time.strftime("%H:%M"),
            snapshot_time.strftime("%A, %b %d %Y"),
            f"{flights_visible:,}",
            f"Avg pred {avg_pred:.1f} min",
            f"{severe_alerts:,}",
            f"Alert precision {alert_precision:.1%}",
            f"{tier_acc:.1%}",
            f"Mean abs err {avg_err:.1f} min",
            fig,
            columns,
            data,
        )

    print("=" * 72)
    print("VALIDATION DAY REPLAY DASHBOARD")
    print("=" * 72)
    print(f"  Date       : {args.date}")
    print(f"  Split      : {args.split}")
    print(f"  CSV        : {args.csv}")
    print(f"  Cache      : {cache_path}")
    print(f"  Threshold  : {args.threshold:.2f}")
    print(f"  URL        : http://127.0.0.1:{args.port}")
    print("=" * 72)
    app.run(debug=False, port=args.port)


if __name__ == "__main__":
    main()
