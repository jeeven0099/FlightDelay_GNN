"""
STEP 13 - EXPORT STATIC WEB DEMO DATA
=====================================
Builds a browser-friendly JSON bundle from the cached replay parquet so the
demo can be hosted as a static site with no live backend.

Usage:
  python 13_export_web_demo_data.py
  python 13_export_web_demo_data.py --date 2021-11-28 --split val
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(r"C:\Users\user\Desktop\Airline_Graphs_Project")
CACHE_DIR = BASE_DIR / "evaluation" / "demo_cache"
WEB_DATA_DIR = BASE_DIR / "web_demo" / "data"
DEFAULT_DATE = "2021-11-28"
DEFAULT_SPLIT = "val"
DEFAULT_THRESHOLD = 0.60


def cache_path(date_str: str, split: str) -> Path:
    return CACHE_DIR / f"demo_replay_{split}_{date_str}.parquet"


def build_export(date_str: str, split: str, severe_threshold: float) -> Path:
    src = cache_path(date_str, split)
    if not src.exists():
        raise FileNotFoundError(f"Replay cache not found: {src}")

    df = pd.read_parquet(src)
    df["snapshot_time"] = pd.to_datetime(df["snapshot_time"])
    df["dep_datetime"] = pd.to_datetime(df["dep_datetime"])
    df["arr_datetime"] = pd.to_datetime(df["arr_datetime"])

    snapshot_times = (
        df[["snapshot_time", "snapshot_label"]]
        .drop_duplicates()
        .sort_values("snapshot_time")
        .reset_index(drop=True)
    )
    snapshot_times["snap_id"] = snapshot_times.index.astype(int)
    df = df.merge(snapshot_times, on="snapshot_time", how="left")

    snapshot_summary = (
        df.groupby("snap_id", as_index=False)
        .agg(
            snapshot_time=("snapshot_time", "first"),
            flights=("flight_id", "count"),
            severe_alerts=("severe_alert", "sum"),
            severe_actuals=("actual_severe", "sum"),
            exact_tier_matches=("tier_match", "sum"),
            mean_abs_err=("abs_err", "mean"),
            mean_pred=("pred", "mean"),
            max_severe_prob=("severe_prob", "max"),
        )
    )
    snapshot_summary["tier_match_rate"] = (
        snapshot_summary["exact_tier_matches"] / snapshot_summary["flights"]
    )
    snapshot_summary["severe_alert_rate"] = (
        snapshot_summary["severe_alerts"] / snapshot_summary["flights"]
    )
    snapshot_summary["snapshot_label"] = pd.to_datetime(
        snapshot_summary["snapshot_time"]
    ).dt.strftime("%Y-%m-%d %H:%M")

    tier_labels = [
        "Early / On Time",
        "Minor (0-15)",
        "Moderate (15-60)",
        "Heavy (60-120)",
        "Severe (120-240)",
        "Extreme (240-720)",
        "Ultra (720+)",
    ]
    horizon_labels = {0: "<1h", 1: "1-3h", 3: "3-6h", 6: ">6h"}
    alert_order = {"TP": 0, "FP": 1, "FN": 2, "TN": 3}
    airport_coords = {}
    for rec in df[["ORIGIN", "origin_lat", "origin_lon"]].drop_duplicates().to_dict(orient="records"):
        if not pd.isna(rec["origin_lat"]) and not pd.isna(rec["origin_lon"]):
            airport_coords[rec["ORIGIN"]] = [round(float(rec["origin_lat"]), 4), round(float(rec["origin_lon"]), 4)]
    for rec in df[["DEST", "dest_lat", "dest_lon"]].drop_duplicates().to_dict(orient="records"):
        if not pd.isna(rec["dest_lat"]) and not pd.isna(rec["dest_lon"]):
            airport_coords.setdefault(rec["DEST"], [round(float(rec["dest_lat"]), 4), round(float(rec["dest_lon"]), 4)])

    rows = []
    keep_cols = [
        "snap_id",
        "flight_id",
        "ORIGIN",
        "DEST",
        "dep_time_label",
        "arr_time_label",
        "Tail_Number",
        "Operating_Airline",
        "horizon_h",
        "hours_to_departure",
        "pred",
        "pred_tier_order",
        "severe_prob",
        "severe_alert",
        "actual",
        "actual_tier_order",
        "abs_err",
        "tier_match",
        "alert_result",
    ]

    for rec in df[keep_cols].sort_values(["snap_id", "pred_tier_order", "pred"], ascending=[True, False, False]).to_dict(orient="records"):
        rows.append(
            {
                "snap": int(rec["snap_id"]),
                "f": int(rec["flight_id"]),
                "o": rec["ORIGIN"],
                "d": rec["DEST"],
                "dep": rec["dep_time_label"],
                "arr": rec["arr_time_label"],
                "tail": rec["Tail_Number"],
                "air": rec["Operating_Airline"],
                "h": int(rec["horizon_h"]),
                "h2d": round(float(rec["hours_to_departure"]), 2),
                "p": round(float(rec["pred"]), 1),
                "po": int(rec["pred_tier_order"]),
                "sp": round(float(rec["severe_prob"]), 3),
                "sa": bool(rec["severe_alert"]),
                "a": round(float(rec["actual"]), 1),
                "ao": int(rec["actual_tier_order"]),
                "ae": round(float(rec["abs_err"]), 1),
                "tm": bool(rec["tier_match"]),
                "ar": rec["alert_result"],
                "alertOrder": alert_order.get(rec["alert_result"], 99),
            }
        )

    payload = {
        "meta": {
            "title": "Flight Delay Replay Demo",
            "date": date_str,
            "split": split,
            "severeThreshold": severe_threshold,
            "rows": len(rows),
            "snapshots": len(snapshot_times),
            "tierLabels": tier_labels,
            "horizonLabels": horizon_labels,
            "airportCoords": airport_coords,
        },
        "snapshotTimes": snapshot_times["snapshot_label"].tolist(),
        "snapshotSummary": [
            {
                "snap": int(rec["snap_id"]),
                "snapshotLabel": rec["snapshot_label"],
                "flights": int(rec["flights"]),
                "severeAlerts": int(rec["severe_alerts"]),
                "severeActuals": int(rec["severe_actuals"]),
                "exactTierMatches": int(rec["exact_tier_matches"]),
                "meanAbsErr": round(float(rec["mean_abs_err"]), 2),
                "meanPred": round(float(rec["mean_pred"]), 2),
                "maxSevereProb": round(float(rec["max_severe_prob"]), 3),
                "tierMatchRate": round(float(rec["tier_match_rate"]), 4),
                "severeAlertRate": round(float(rec["severe_alert_rate"]), 4),
            }
            for rec in snapshot_summary.to_dict(orient="records")
        ],
        "rowsData": rows,
    }

    WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = WEB_DATA_DIR / f"demo_{split}_{date_str}.json"
    out_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=DEFAULT_DATE)
    parser.add_argument("--split", default=DEFAULT_SPLIT, choices=["val", "test"])
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()

    out_path = build_export(args.date, args.split, args.threshold)
    print(f"Static demo JSON written to {out_path}")
    print(f"Size: {out_path.stat().st_size / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    main()
