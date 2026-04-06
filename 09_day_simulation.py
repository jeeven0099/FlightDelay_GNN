"""
STEP 9 — DAY SIMULATION (FLIGHT TRACKING)
==========================================
For every flight on a given test date, tracks the model's prediction
at three points in time:
  - 6 hours before departure  (Low confidence — earliest warning)
  - 3 hours before departure  (Medium confidence)
  - 1 hour before departure   (High confidence)

Then compares all three against the actual ArrDelay.

NOTE: The model predicts ARRIVAL DELAY only.
      Departure delay is an input feature, not an output.

USAGE:
  python 09_day_simulation.py --date 2022-11-04
  python 09_day_simulation.py --date 2022-11-04 --save_plots

OUTPUT:
  outputs/flight_tracking_YYYY-MM-DD.csv
      One row per flight with columns:
      flight_id, ORIGIN, DEST, dep_time, arr_time,
      pred_6h, pred_3h, pred_1h, actual_arr_delay,
      error_6h, error_3h, error_1h,
      caught_at_6h, caught_at_3h, caught_at_1h

  outputs/day_simulation_YYYY-MM-DD_report.txt
      Summary statistics and comparison with published baselines
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from importlib.util import spec_from_file_location, module_from_spec
from collections import defaultdict

# ── CONFIG ──────────────────────────────────────────────────────────────────
BASE_DIR        = r"C:\Users\user\Desktop\Airline_Graphs_Project"
GRAPH_DATA_DIR  = os.path.join(BASE_DIR, "graph_data")
CHECKPOINT_DIR  = os.path.join(BASE_DIR, "checkpoints")
OUTPUT_DIR      = os.path.join(BASE_DIR, "outputs")
DELAY_THRESHOLD = 15.0   # DOT standard definition of "delayed"
WINDOW_HOURS    = 8      # must match step 5
# ────────────────────────────────────────────────────────────────────────────


def load_dashboard():
    path = os.path.join(BASE_DIR, "07_dashboard.py")
    spec = spec_from_file_location("dashboard", path)
    mod  = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_simulation(date_str, device, save_plots=False):
    print("=" * 65)
    print(f"DAY SIMULATION — {date_str}")
    print(f"Tracks each flight at 6h / 3h / 1h before departure")
    print("=" * 65)

    # ── Load everything ──────────────────────────────────────────────────────
    print("\nLoading model ...")
    dash      = load_dashboard()
    ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    model     = dash.load_model(ckpt_path, device)
    model.eval()

    static_edges = torch.load(
        os.path.join(GRAPH_DATA_DIR, "static_edges.pt"),
        map_location=device, weights_only=False)

    airport_index = pd.read_parquet(
        os.path.join(GRAPH_DATA_DIR, "airport_index.parquet"))
    airports   = airport_index["airport"].tolist()
    ap_idx_map = {ap: i for i, ap in enumerate(airports)}

    print("Loading flight lookup ...")
    flight_lookup = pd.read_parquet(
        os.path.join(GRAPH_DATA_DIR, "flight_lookup.parquet"))
    flight_lookup["dep_datetime"] = pd.to_datetime(
        flight_lookup["dep_datetime"], errors="coerce")
    flight_lookup["arr_datetime"] = pd.to_datetime(
        flight_lookup["arr_datetime"], errors="coerce")

    # Load route stats for feature building (features 17-18)
    route_stats_lookup = None
    rs_path = os.path.join(GRAPH_DATA_DIR, "route_stats.parquet")
    rs_global_path = os.path.join(GRAPH_DATA_DIR, "route_stats_global.parquet")
    if os.path.exists(rs_path):
        try:
            rs = pd.read_parquet(rs_path)
            rg = pd.read_parquet(rs_global_path)
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
            route_stats_lookup = (lf, lh, lr, gm, gs)
            print(f"  ✓ Route stats loaded ({len(lf):,} route×hour×dow entries)")
        except Exception as e:
            print(f"  ⚠ Route stats not loaded: {e}")

    # Load ArrDelay from weather_node_features.parquet — same 36-airport dataset
    has_arr_delay = False
    wx_path  = os.path.join(GRAPH_DATA_DIR, "weather_node_features.parquet")
    raw_path = os.path.join(BASE_DIR, "flights_2018_2022.parquet")

    src_path = wx_path if os.path.exists(wx_path) else \
               raw_path if os.path.exists(raw_path) else None

    if src_path:
        print(f"  Loading ArrDelay from {os.path.basename(src_path)} ...")
        try:
            src = pd.read_parquet(src_path,
                                  columns=["ORIGIN","DEST",
                                           "dep_datetime","ArrDelay"])
            src["dep_datetime"] = pd.to_datetime(
                src["dep_datetime"], errors="coerce")

            # Merge into flight_lookup on exact ORIGIN + DEST + dep_datetime
            flight_lookup = flight_lookup.merge(
                src[["ORIGIN","DEST","dep_datetime","ArrDelay"]],
                on=["ORIGIN","DEST","dep_datetime"],
                how="left")

            matched   = flight_lookup["ArrDelay"].notna().sum()
            match_pct = matched / len(flight_lookup) * 100
            print(f"  ✓ ArrDelay matched for {matched:,} / "
                  f"{len(flight_lookup):,} flights ({match_pct:.0f}%)")
            has_arr_delay = matched > 0
        except Exception as e:
            print(f"  ⚠ Could not load ArrDelay: {e}")
    else:
        print(f"  ⚠ No source parquet found — prediction-only mode")

    print("Loading test snapshots ...")
    test_snaps = torch.load(
        os.path.join(GRAPH_DATA_DIR, "snapshots_test.pt"),
        map_location=device, weights_only=False)

    # ── Get all snapshots for target date ────────────────────────────────────
    target_date = pd.Timestamp(date_str).date()
    day_snaps   = sorted(
        [s for s in test_snaps
         if pd.Timestamp(s["airport"].snapshot_time).date() == target_date],
        key=lambda s: pd.Timestamp(s["airport"].snapshot_time))

    if not day_snaps:
        print(f"\n❌ No snapshots for {date_str}")
        avail = sorted(set(
            str(pd.Timestamp(s["airport"].snapshot_time).date())
            for s in test_snaps))
        print(f"Available dates: {avail[:5]} ... {avail[-5:]}")
        return None, None

    print(f"  ✓ {len(day_snaps)} snapshots for {date_str}")

    # ── Load previous day snapshots for GRU warm-up ──────────────────────────
    # Flights departing 00:00-06:00 need snapshots from the previous evening
    # to get their 6h/3h/1h predictions. We warm up the GRU on the last 6
    # snapshots of the previous day — exactly as a live system would do.
    prev_date   = (pd.Timestamp(date_str) - pd.Timedelta(days=1)).date()
    prev_snaps  = sorted(
        [s for s in test_snaps
         if pd.Timestamp(s["airport"].snapshot_time).date() == prev_date],
        key=lambda s: pd.Timestamp(s["airport"].snapshot_time))

    WARMUP_HOURS = 6
    warmup_snaps = prev_snaps[-WARMUP_HOURS:] if prev_snaps else []

    if warmup_snaps:
        print(f"  ✓ {len(warmup_snaps)} warm-up snapshots from {prev_date} "
              f"(last {WARMUP_HOURS}h of previous day)")
    else:
        print(f"  ⚠ No previous-day snapshots found for {prev_date}")
        print(f"    Midnight flights will lack 6h/3h predictions")

    # ── Get all flights for this date ────────────────────────────────────────
    date_flights = flight_lookup[
        flight_lookup["dep_datetime"].dt.date == target_date
    ].copy().reset_index(drop=True)
    print(f"  ✓ {len(date_flights):,} flights on {date_str}")

    if len(date_flights) == 0:
        print("❌ No flights found for this date in flight_lookup")
        return None, None

    # ── Static edges to device ───────────────────────────────────────────────
    cg_ei = static_edges["congestion_ei"].to(device)
    cg_ea = (static_edges["congestion_ea"].to(device)
             if "congestion_ea" in static_edges
             else torch.zeros((0, 1), dtype=torch.float, device=device))
    nw_ei = static_edges["network_ei"].to(device)
    nw_ea = static_edges["network_ea"].to(device)

    # ── ALL snapshots: warmup + target day ───────────────────────────────────
    # warmup snapshots are indexed as negative numbers (-6 to -1)
    # target day snapshots are indexed 0..23
    all_snaps   = warmup_snaps + day_snaps
    # snap_times covers both warmup and target day
    all_times   = [pd.Timestamp(s["airport"].snapshot_time)
                   for s in all_snaps]
    # Index offset: warmup snapshots are 0..len(warmup)-1 in all_snaps
    # target day snapshots start at offset = len(warmup_snaps)
    day_offset  = len(warmup_snaps)
    snap_times  = all_times   # full list including warmup

    def find_snap_idx(dep_dt, hours_before):
        """Find index in all_snaps closest to dep_dt - hours_before."""
        target = dep_dt - pd.Timedelta(hours=hours_before)
        diffs  = [abs((t - target).total_seconds()) for t in snap_times]
        best   = int(np.argmin(diffs))
        # Only accept if within 35 minutes of target (hourly snapshots)
        if diffs[best] < 35 * 60:
            return best
        return None

    # ── Run model — warmup first, then target day ─────────────────────────────
    total = len(all_snaps)
    print(f"\nRunning model across {len(warmup_snaps)} warm-up + "
          f"{len(day_snaps)} target snapshots ...")
    if warmup_snaps:
        print(f"  Warm-up: {all_times[0].strftime('%H:%M')} → "
              f"{all_times[day_offset-1].strftime('%H:%M')} "
              f"(builds GRU state from previous evening)")
    print(f"  Target:  00:00 → 23:00 on {date_str}\n")
    print(f"{'Hour':>8}  {'Phase':>10}  {'Flights in window':>18}  "
          f"{'Avg pred delay':>16}")
    print(f"{'─'*58}")

    snap_predictions = {}
    ap_h = model.init_hidden(torch.device(device))

    with torch.no_grad():
        for si, snap in enumerate(all_snaps):
            is_warmup  = si < day_offset
            phase      = "warm-up" if is_warmup else "predict"
            snap       = snap.to(device)
            snap_time  = all_times[si]

            snap["airport","congestion","airport"].edge_index = cg_ei
            snap["airport","congestion","airport"].edge_attr  = cg_ea
            snap["airport","network",   "airport"].edge_index = nw_ei
            snap["airport","network",   "airport"].edge_attr  = nw_ea

            window_end  = snap_time + pd.Timedelta(hours=WINDOW_HOURS)
            if is_warmup:
                flights_now = date_flights[
                    date_flights["dep_datetime"].dt.hour.between(
                        snap_time.hour, min(snap_time.hour + WINDOW_HOURS, 23))
                ].head(500).reset_index(drop=True)
            else:
                flights_now = date_flights[
                    (date_flights["dep_datetime"] >= snap_time) &
                    (date_flights["dep_datetime"] <  window_end)
                ].reset_index(drop=True)

            if snap["flight"].num_nodes == 0 and len(flights_now) > 0:
                snap = _build_flight_nodes(
                    snap, flights_now, snap_time, ap_idx_map, device,
                    route_stats_lookup=route_stats_lookup)
            elif snap["flight"].num_nodes == 0:
                snap["flight"].x         = torch.zeros(
                    (0,19), dtype=torch.float16).to(device)
                snap["flight"].num_nodes = 0
                for et in [("flight","departs_from","airport"),
                           ("flight","arrives_at","airport"),
                           ("flight","rotation","flight")]:
                    snap[et].edge_index = torch.zeros(
                        (2,0), dtype=torch.long).to(device)
                    snap[et].edge_attr  = torch.zeros(
                        (0,1), dtype=torch.float).to(device)

            ap_pred, fl_pred, fl_logits, ap_h = model(snap, ap_h)
            ap_h = ap_h.detach()

            n_fl = snap["flight"].num_nodes
            preds_this_snap = {}
            if not is_warmup and n_fl > 0 and len(fl_pred) > 0:
                fl_preds_np = fl_pred.cpu().numpy()
                for i in range(min(n_fl, len(fl_preds_np), len(flights_now))):
                    fr  = flights_now.iloc[i]
                    key = _flight_key(fr)
                    preds_this_snap[key] = float(fl_preds_np[i])

            snap_predictions[si] = preds_this_snap

            avg_pred = np.mean(list(preds_this_snap.values())) \
                       if preds_this_snap else float("nan")
            avg_str  = "—" if np.isnan(avg_pred) else f"{avg_pred:+.1f} min"
            print(f"  {snap_time.strftime('%H:%M')}  "
                  f"{phase:>10}  "
                  f"{len(flights_now):>18,}  "
                  f"{avg_str:>16}")

    # ── Build per-flight tracking table ─────────────────────────────────────
    print(f"\nBuilding per-flight prediction table ...")

    rows = []
    for _, fl in date_flights.iterrows():
        dep_dt = fl.get("dep_datetime")
        arr_dt = fl.get("arr_datetime")
        if dep_dt is None or not pd.notna(dep_dt):
            continue

        key = _flight_key(fl)

        # Get actual arrival delay — merged from source parquet
        actual = None
        if has_arr_delay:
            v = fl.get("ArrDelay")
            if v is not None and pd.notna(v):
                actual = float(v)

        # Look up prediction at each horizon
        pred_6h = pred_3h = pred_1h = None

        for hours_before, label in [(6, "pred_6h"), (3, "pred_3h"), (1, "pred_1h")]:
            si = find_snap_idx(dep_dt, hours_before)
            if si is not None and key in snap_predictions.get(si, {}):
                if label == "pred_6h": pred_6h = snap_predictions[si][key]
                if label == "pred_3h": pred_3h = snap_predictions[si][key]
                if label == "pred_1h": pred_1h = snap_predictions[si][key]

        # Skip flights we couldn't get any prediction for
        if pred_6h is None and pred_3h is None and pred_1h is None:
            continue

        row = {
            "ORIGIN"       : fl.get("ORIGIN",""),
            "DEST"         : fl.get("DEST",""),
            "Airline"      : fl.get("Operating_Airline",""),
            "Tail"         : fl.get("Tail_Number",""),
            "Sched Dep"    : dep_dt.strftime("%H:%M") if pd.notna(dep_dt) else "—",
            "Sched Arr"    : arr_dt.strftime("%H:%M")
                             if arr_dt is not None and pd.notna(arr_dt) else "—",
            "Pred 6h (min)": round(pred_6h, 1) if pred_6h is not None else None,
            "Pred 3h (min)": round(pred_3h, 1) if pred_3h is not None else None,
            "Pred 1h (min)": round(pred_1h, 1) if pred_1h is not None else None,
            "Actual ArrDelay": round(actual, 1) if actual is not None else None,
        }

        # Errors
        for p, col in [(pred_6h,"Error 6h"), (pred_3h,"Error 3h"),
                       (pred_1h,"Error 1h")]:
            row[col] = round(abs(p - actual), 1) \
                       if p is not None and actual is not None else None

        # Was it correctly flagged as delayed?
        was_delayed = actual >= DELAY_THRESHOLD if actual is not None else None
        for p, col in [(pred_6h,"Caught 6h"), (pred_3h,"Caught 3h"),
                       (pred_1h,"Caught 1h")]:
            if p is not None and was_delayed is not None:
                row[col] = (p >= DELAY_THRESHOLD) == was_delayed
            else:
                row[col] = None

        # Status label
        if actual is not None:
            if actual >= 60:   row["Status"] = "🔴 Severe"
            elif actual >= 30: row["Status"] = "🟠 Moderate"
            elif actual >= 15: row["Status"] = "🟡 Minor"
            elif actual >= 0:  row["Status"] = "🟢 On Time"
            else:              row["Status"] = "🔵 Early"
        else:
            row["Status"] = "—"

        rows.append(row)

    flight_df = pd.DataFrame(rows)
    print(f"  ✓ {len(flight_df):,} flights tracked")

    # ── Print summary ────────────────────────────────────────────────────────
    _print_summary(flight_df, date_str)

    # ── Save outputs ─────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, f"flight_tracking_{date_str}.csv")
    flight_df.to_csv(csv_path, index=False)
    print(f"\n  ✅ Full flight table → {csv_path}")
    print(f"     Open in Excel to explore all {len(flight_df):,} flights")

    if save_plots:
        _make_plots(flight_df, date_str)

    return flight_df


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _flight_key(row):
    """Unique key for a flight: origin + dest + departure hour."""
    dep = row.get("dep_datetime")
    dep_h = dep.hour if dep is not None and pd.notna(dep) else 0
    return (str(row.get("ORIGIN","")),
            str(row.get("DEST","")),
            dep_h)


def _build_flight_nodes(snap, flights_w, snap_time, ap_idx_map, device,
                         route_stats_lookup=None):
    """Build flight feature matrix (19 dims) from flight_lookup data.
    Applies same horizon masking as training."""
    n_fl       = len(flights_w)
    X          = np.zeros((n_fl, 19), dtype=np.float32)
    snap_naive = snap_time.replace(tzinfo=None) \
                 if hasattr(snap_time,"tzinfo") else snap_time

    MAX_DELAY_MIN        = 300.0
    MASK_FULL_THRESHOLD  = 2.0 / 24
    MASK_PART_THRESHOLD  = 1.0 / 24

    for i, (_, row) in enumerate(flights_w.iterrows()):
        dep_dt = row.get("dep_datetime")
        arr_dt = row.get("arr_datetime")
        origin = row.get("ORIGIN","")
        dest   = row.get("DEST","")

        if dep_dt is not None and pd.notna(dep_dt):
            h2dep    = max(0.0, (dep_dt - snap_naive).total_seconds()/3600)
            dep_hour = dep_dt.hour
            arr_hour = arr_dt.hour if arr_dt is not None and pd.notna(arr_dt) else 0
            dow      = dep_dt.dayofweek
        else:
            h2dep = 1.0; dep_hour = 0; arr_hour = 0; dow = 0

        time_to_dep = min(h2dep / 24.0, 1.0)

        # Schedule features — always available
        X[i, 1]  = np.sin(2*np.pi*dep_hour/24)
        X[i, 2]  = np.cos(2*np.pi*dep_hour/24)
        X[i, 6]  = 1.0
        X[i, 8]  = np.sin(2*np.pi*dow/7)
        X[i, 9]  = np.cos(2*np.pi*dow/7)
        X[i,10]  = 1.0 if origin in ap_idx_map else 0.0
        X[i,12]  = np.sin(2*np.pi*arr_hour/24)
        X[i,13]  = np.cos(2*np.pi*arr_hour/24)
        X[i,14]  = time_to_dep

        # Horizon masking — matches training exactly
        # Gate features (0,3,4,7,11) zeroed for far-out flights
        if time_to_dep >= MASK_FULL_THRESHOLD:
            pass   # all gate features stay 0
        elif time_to_dep >= MASK_PART_THRESHOLD:
            pass   # dep_delay + taxi still 0, immed may be known
        # else < 1h: gate features could be set if available

        # Route stats — never masked, always available
        if route_stats_lookup is not None:
            lf, lh, lr, gm, gs = route_stats_lookup
            key_f = (origin, dest, dep_hour, int(dow))
            key_h = (origin, dest, dep_hour)
            key_r = (origin, dest)
            if   key_f in lf: h_avg, h_std = lf[key_f]
            elif key_h in lh: h_avg, h_std = lh[key_h]
            elif key_r in lr: h_avg, h_std = lr[key_r]
            else:             h_avg, h_std = gm, gs
            X[i,17] = np.clip(h_avg / MAX_DELAY_MIN, -1, 1)
            X[i,18] = np.clip(h_std / MAX_DELAY_MIN,  0, 1)

    snap["flight"].x         = torch.tensor(X, dtype=torch.float16).to(device)
    snap["flight"].num_nodes = n_fl

    if "ArrDelay" in flights_w.columns:
        y    = flights_w["ArrDelay"].fillna(0).values.astype(np.float32)
        mask = flights_w["ArrDelay"].notna().values
    else:
        y    = np.zeros(n_fl, dtype=np.float32)
        mask = np.zeros(n_fl, dtype=bool)
    snap["flight"].y      = torch.tensor(y,    dtype=torch.float).to(device)
    snap["flight"].y_mask = torch.tensor(mask, dtype=torch.bool).to(device)

    dep_src = list(range(n_fl))
    dep_dst = [ap_idx_map.get(flights_w.iloc[i].get("ORIGIN",""), 0)
               for i in range(n_fl)]
    ei = torch.tensor([dep_src, dep_dst], dtype=torch.long).to(device)
    ea = torch.ones((n_fl,1), dtype=torch.float).to(device)
    snap["flight","departs_from","airport"].edge_index = ei
    snap["flight","departs_from","airport"].edge_attr  = ea
    snap["flight","arrives_at",  "airport"].edge_index = \
        torch.zeros((2,0), dtype=torch.long).to(device)
    snap["flight","arrives_at",  "airport"].edge_attr  = \
        torch.zeros((0,1), dtype=torch.float).to(device)
    snap["flight","rotation",    "flight" ].edge_index = \
        torch.zeros((2,0), dtype=torch.long).to(device)
    snap["flight","rotation",    "flight" ].edge_attr  = \
        torch.zeros((0,2), dtype=torch.float).to(device)
    return snap


def _print_summary(df, date_str):
    """Print the full simulation report."""
    print(f"\n{'='*65}")
    print(f"RESULTS — {date_str}")
    print(f"{'='*65}")
    print(f"\nModel predicts: ARRIVAL DELAY (minutes late at destination)")
    print(f"Baseline:       Dep delay used as INPUT feature, not output")

    has_actual = df["Actual ArrDelay"].notna().any()

    if not has_actual:
        print(f"\n  ⚠ ArrDelay not available in flight_lookup for this date")
        print(f"    The CSV has predictions at all three horizons — no actuals to compare")
        print(f"\n  Prediction sample (first 20 flights with 1h predictions):")
        print(f"  {'Route':>10}  {'Dep':>6}  "
              f"{'Pred 6h':>8}  {'Pred 3h':>8}  {'Pred 1h':>8}  {'Note':>20}")
        print(f"  {'─'*65}")
        shown = 0
        for _, r in df.iterrows():
            p1 = r['Pred 1h (min)']
            if p1 is None or (isinstance(p1, float) and np.isnan(p1)):
                continue  # skip flights with no predictions at any horizon
            route = f"{r['ORIGIN']}→{r['DEST']}"

            def fmt(v):
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return "     —"
                return f"{v:>+6.0f}m"

            p6 = fmt(r['Pred 6h (min)'])
            p3 = fmt(r['Pred 3h (min)'])
            p1s = fmt(r['Pred 1h (min)'])

            # Note why 6h/3h may be missing
            note = ""
            if r['Pred 6h (min)'] is None or \
                    (isinstance(r['Pred 6h (min)'], float) and
                     np.isnan(r['Pred 6h (min)'])):
                note = "← early dep, no 6h snap"
            elif r['Pred 3h (min)'] is None or \
                    (isinstance(r['Pred 3h (min)'], float) and
                     np.isnan(r['Pred 3h (min)'])):
                note = "← no 3h snap"

            print(f"  {route:>10}  {r['Sched Dep']:>6}  "
                  f"{p6}  {p3}  {p1s}  {note}")
            shown += 1
            if shown >= 20:
                break

        # Also show flights WITH 6h predictions
        has_6h = df[df['Pred 6h (min)'].notna() &
                    df['Pred 6h (min)'].apply(
                        lambda x: not (isinstance(x, float) and np.isnan(x)))]
        print(f"\n  Flights with all 3 predictions: {len(has_6h):,} "
              f"({len(has_6h)/len(df)*100:.1f}%)")
        if len(has_6h) > 0:
            print(f"\n  Sample with all 3 horizons:")
            print(f"  {'Route':>10}  {'Dep':>6}  "
                  f"{'Pred 6h':>8}  {'Pred 3h':>8}  {'Pred 1h':>8}")
            print(f"  {'─'*50}")
            for _, r in has_6h.head(10).iterrows():
                route = f"{r['ORIGIN']}→{r['DEST']}"
                print(f"  {route:>10}  {r['Sched Dep']:>6}  "
                      f"{r['Pred 6h (min)']:>+7.0f}m  "
                      f"{r['Pred 3h (min)']:>+7.0f}m  "
                      f"{r['Pred 1h (min)']:>+7.0f}m")

    # With actuals
    eval_df = df[df["Actual ArrDelay"].notna()].copy()
    n       = len(eval_df)

    print(f"\n  Flights with actual outcomes: {n:,}")
    print(f"\n  PER-HORIZON ACCURACY")
    print(f"  {'─'*60}")
    print(f"  {'Horizon':>10}  {'Flights':>8}  {'MAE':>8}  "
          f"{'RMSE':>8}  {'Delay Acc':>10}  {'Bias':>8}")
    print(f"  {'─'*60}")

    for col, h in [("Error 6h","6h"),("Error 3h","3h"),("Error 1h","1h")]:
        pred_col  = f"Pred {h[:1]}h (min)" if h != "6h" else "Pred 6h (min)"
        pred_col  = {"6h":"Pred 6h (min)","3h":"Pred 3h (min)",
                     "1h":"Pred 1h (min)"}[h]
        caught_col = {"6h":"Caught 6h","3h":"Caught 3h","1h":"Caught 1h"}[h]

        sub = eval_df[eval_df[col].notna()].copy()
        if len(sub) == 0:
            print(f"  {h:>10}:  no data")
            continue

        mae   = sub[col].mean()
        rmse  = np.sqrt((sub[col]**2).mean())
        dacc  = sub[caught_col].mean() * 100 if caught_col in sub else float("nan")
        bias  = (sub[pred_col] - sub["Actual ArrDelay"]).mean()

        print(f"  {h:>10}  {len(sub):>8,}  {mae:>8.2f}  "
              f"{rmse:>8.2f}  {dacc:>9.1f}%  {bias:>+8.2f}")

    # Baseline comparison
    best_col = "Error 1h" if eval_df["Error 1h"].notna().any() else "Error 3h"
    best_mae = eval_df[eval_df[best_col].notna()][best_col].mean()
    print(f"\n  COMPARISON WITH PUBLISHED BASELINES")
    print(f"  {'─'*55}")
    print(f"  {'Method':<40}  {'MAE':>6}  {'Horizon':>8}")
    print(f"  {'─'*55}")
    print(f"  {'UC Berkeley XGBoost (2025)':<40}  {'12.79':>6}  {'~0h':>8}")
    print(f"  {'FDPP-ML (2023)':<40}  {'16.70':>6}  {'2h':>8}")
    print(f"  {'Your model (this day, 1h)':<40}  {best_mae:>6.2f}  {'1h':>8}")

    # Status distribution
    status_counts = eval_df["Status"].value_counts()
    print(f"\n  ACTUAL DELAY DISTRIBUTION ON {date_str}")
    print(f"  {'─'*35}")
    for status, count in status_counts.items():
        pct = count / n * 100
        print(f"  {status:<20} {count:>6,}  ({pct:.1f}%)")

    # Best catches — severe delays the model predicted 6h early
    severe = eval_df[eval_df["Actual ArrDelay"] >= 30].copy()
    if len(severe) > 0 and "Pred 6h (min)" in severe.columns:
        caught_6h = severe[severe["Pred 6h (min)"] >= DELAY_THRESHOLD]
        print(f"\n  SEVERE DELAYS (≥30min) CAUGHT AT 6H AHEAD")
        print(f"  Catch rate: {len(caught_6h)}/{len(severe)} "
              f"({len(caught_6h)/len(severe)*100:.1f}%)")
        if len(caught_6h) > 0:
            print(f"\n  {'Route':>10}  {'Dep':>6}  "
                  f"{'6h Pred':>8}  {'3h Pred':>8}  "
                  f"{'1h Pred':>8}  {'Actual':>8}")
            print(f"  {'─'*60}")
            for _, r in caught_6h.head(10).iterrows():
                route = f"{r['ORIGIN']}→{r['DEST']}"
                print(f"  {route:>10}  {r['Sched Dep']:>6}  "
                      f"{r['Pred 6h (min)']:>+7.0f}m  "
                      f"{r['Pred 3h (min)']:>+7.0f}m  "
                      f"{r['Pred 1h (min)']:>+7.0f}m  "
                      f"{r['Actual ArrDelay']:>+7.0f}m")

    # Worst misses
    worst = eval_df[eval_df["Error 1h"].notna()].nlargest(10, "Error 1h")
    if len(worst) > 0:
        print(f"\n  TOP 10 WORST MISSES (1h prediction)")
        print(f"  {'Route':>10}  {'Dep':>6}  "
              f"{'6h Pred':>8}  {'3h Pred':>8}  "
              f"{'1h Pred':>8}  {'Actual':>8}  {'Error':>7}")
        print(f"  {'─'*68}")
        for _, r in worst.iterrows():
            route = f"{r['ORIGIN']}→{r['DEST']}"
            print(f"  {route:>10}  {r['Sched Dep']:>6}  "
                  f"{r['Pred 6h (min)']:>+7.0f}m  "
                  if pd.notna(r['Pred 6h (min)']) else
                  f"  {route:>10}  {r['Sched Dep']:>6}  {'—':>8}  ", end="")
            for col in ["Pred 6h (min)","Pred 3h (min)","Pred 1h (min)"]:
                val = r[col]
                print(f"{val:>+7.0f}m  " if pd.notna(val) else "     —   ", end="")
            print(f"{r['Actual ArrDelay']:>+7.0f}m  {r['Error 1h']:>6.0f}m")


def _make_plots(df, date_str):
    """Generate summary plots if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        eval_df = df[df["Actual ArrDelay"].notna()].copy()
        if len(eval_df) == 0:
            print("  ⚠ No actual delays to plot")
            return

        fig = plt.figure(figsize=(18, 10))
        fig.suptitle(f"Flight Delay Prediction — {date_str}\n"
                     f"6h / 3h / 1h Before Departure vs Actual ArrDelay",
                     fontsize=13, fontweight="bold")
        gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

        colors = {"6h":"#9b59b6","3h":"#e67e22","1h":"#2ecc71"}
        lim = min(120, max(
            eval_df["Actual ArrDelay"].abs().quantile(0.95), 30))

        # Scatter plots: pred vs actual at each horizon
        for col_i, (h, pred_col) in enumerate([
                ("6h","Pred 6h (min)"),
                ("3h","Pred 3h (min)"),
                ("1h","Pred 1h (min)")]):
            ax = fig.add_subplot(gs[0, col_i])
            sub = eval_df[eval_df[pred_col].notna()].sample(
                min(3000, len(eval_df)), random_state=42)
            ax.scatter(sub["Actual ArrDelay"], sub[pred_col],
                       alpha=0.15, s=6, color=colors[h])
            ax.plot([-lim,lim],[-lim,lim],"r--",alpha=0.5,lw=1)
            mae = eval_df[eval_df[pred_col].notna()][
                f"Error {h}"].mean()
            ax.set_title(f"{h} Before Departure\nMAE = {mae:.2f} min",
                         fontsize=10)
            ax.set_xlabel("Actual ArrDelay (min)", fontsize=8)
            ax.set_ylabel("Predicted ArrDelay (min)", fontsize=8)
            ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim)
            ax.axhline(0,color="gray",alpha=0.3,lw=0.5)
            ax.axvline(0,color="gray",alpha=0.3,lw=0.5)

        # MAE comparison bar
        ax4 = fig.add_subplot(gs[0, 3])
        maes   = []
        labels = []
        for h, pred_col in [("6h","Pred 6h (min)"),
                             ("3h","Pred 3h (min)"),
                             ("1h","Pred 1h (min)")]:
            sub = eval_df[eval_df[pred_col].notna()]
            if len(sub) > 0:
                maes.append(sub[f"Error {h}"].mean())
                labels.append(f"{h}\n({len(sub):,} flights)")
        bars = ax4.bar(labels, maes,
                       color=[colors[h[:2]] for h in ["6h","3h","1h"]],
                       alpha=0.85)
        ax4.axhline(12.79, color="#e74c3c", linestyle="--",
                    alpha=0.7, label="Berkeley XGB (12.79)")
        ax4.axhline(9.68,  color="#2ecc71", linestyle="--",
                    alpha=0.7, label="Test set avg (9.68)")
        for bar, mae in zip(bars, maes):
            ax4.text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()+0.1,
                     f"{mae:.2f}", ha="center", fontsize=9)
        ax4.set_ylabel("MAE (minutes)")
        ax4.set_title("MAE by Horizon")
        ax4.legend(fontsize=7)
        ax4.set_ylim(0, max(maes + [14]) * 1.2)

        # Error distribution by horizon
        ax5 = fig.add_subplot(gs[1, :2])
        for h, pred_col, col in [
                ("6h","Pred 6h (min)","Error 6h"),
                ("3h","Pred 3h (min)","Error 3h"),
                ("1h","Pred 1h (min)","Error 1h")]:
            sub = eval_df[eval_df[pred_col].notna()]
            errors = (sub[pred_col] - sub["Actual ArrDelay"]).clip(-60,60)
            ax5.hist(errors, bins=60, alpha=0.45, label=f"{h} (MAE={sub[col].mean():.1f})",
                     color=colors[h], density=True)
        ax5.axvline(0, color="red", linestyle="--", alpha=0.6, lw=1)
        ax5.set_xlabel("Prediction Error (min)")
        ax5.set_ylabel("Density")
        ax5.set_title("Error Distribution by Horizon")
        ax5.legend(fontsize=9)

        # Prediction convergence — how predictions change as we get closer
        ax6 = fig.add_subplot(gs[1, 2:])
        # For flights with all 3 predictions, show how error evolves
        complete = eval_df[
            eval_df["Error 6h"].notna() &
            eval_df["Error 3h"].notna() &
            eval_df["Error 1h"].notna()
        ]
        if len(complete) > 50:
            x = [6, 3, 1]
            # Delayed flights
            delayed = complete[complete["Actual ArrDelay"] >= DELAY_THRESHOLD]
            ontime  = complete[complete["Actual ArrDelay"] < DELAY_THRESHOLD]
            if len(delayed) > 0:
                d_maes = [delayed["Error 6h"].mean(),
                          delayed["Error 3h"].mean(),
                          delayed["Error 1h"].mean()]
                ax6.plot(x, d_maes, "o-", color="#e74c3c",
                         label=f"Delayed flights (n={len(delayed):,})", lw=2)
            if len(ontime) > 0:
                o_maes = [ontime["Error 6h"].mean(),
                          ontime["Error 3h"].mean(),
                          ontime["Error 1h"].mean()]
                ax6.plot(x, o_maes, "o-", color="#2ecc71",
                         label=f"On-time flights (n={len(ontime):,})", lw=2)
            ax6.axhline(12.79, color="gray", linestyle="--",
                        alpha=0.5, label="Berkeley baseline")
            ax6.set_xlabel("Hours Before Departure")
            ax6.set_ylabel("MAE (minutes)")
            ax6.set_title("Prediction Accuracy vs Time to Departure\n"
                          "(Does accuracy improve as we get closer?)")
            ax6.set_xticks([1,3,6])
            ax6.invert_xaxis()
            ax6.legend(fontsize=9)

        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR,
                                 f"day_simulation_{date_str}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✅ Plot saved → {plot_path}")

    except ImportError:
        print("  pip install matplotlib  to get plots")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="2022-11-04",
                        help="Test date (YYYY-MM-DD). "
                             "Busiest available: 2022-11-04, 2022-08-04, "
                             "2022-11-02, 2022-10-04")
    parser.add_argument("--save_plots", action="store_true",
                        help="Save matplotlib plots (pip install matplotlib)")
    args = parser.parse_args()

    device = torch.device("cpu")
    run_simulation(args.date, device, save_plots=args.save_plots)


if __name__ == "__main__":
    main()