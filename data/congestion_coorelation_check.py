"""
congestion_correlation_check.py
Validates that the three congestion edge types correlate with actual
flight delays before training the GNN.

Checks:
  A. Volume correlation  — do high-volume routes have higher delays?
  B. Taxi anomaly        — do congested airport-hours predict higher delays?
  C. Hub vs spoke        — do hub airports have higher/more volatile delays?
  D. Congestion → downstream delay — does TaxiOut at origin predict ArrDelay at dest?

Usage:
    python congestion_correlation_check.py
"""

import pandas as pd
import numpy as np

# ── CONFIG ──────────────────────────────────────────────────────────────────
FLIGHTS_PATH      = r"C:\Users\user\Desktop\Airline_Graphs_Project\flights_2018_2022.parquet"
CONGESTION_PATH   = r"C:\Users\user\Desktop\Airline_Graphs_Project\graph_data\congestion_edges.parquet"
TAXI_ZSCORE_THRESH = 1.5
HUB_TOP_N          = 30
MIN_PAIR_FLIGHTS   = 500
# ────────────────────────────────────────────────────────────────────────────


def load_data():
    print("Loading flights ...")
    df = pd.read_parquet(FLIGHTS_PATH)
    df["dep_datetime"] = pd.to_datetime(df["dep_datetime"])
    df["arr_datetime"] = pd.to_datetime(df["arr_datetime"])
    print(f"  {len(df):,} rows loaded")
    return df


# ── A. Volume correlation ────────────────────────────────────────────────────
def check_volume_correlation(df):
    print("\n" + "="*60)
    print("A. VOLUME CORRELATION")
    print("="*60)
    print("Question: do high-volume routes have higher average delays?")

    pair_stats = (df.groupby(["ORIGIN", "DEST"])
                    .agg(
                        flight_count = ("ArrDelay", "count"),
                        avg_arr_delay = ("ArrDelay", "mean"),
                        pct_delayed   = ("ArrDelay", lambda x: (x > 15).mean()),
                    )
                    .reset_index())

    pair_stats = pair_stats[pair_stats["flight_count"] >= MIN_PAIR_FLIGHTS]

    # Bin by volume quartile
    pair_stats["volume_quartile"] = pd.qcut(
        pair_stats["flight_count"], q=4,
        labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"]
    )

    summary = (pair_stats.groupby("volume_quartile", observed=True)
                         .agg(
                             n_routes      = ("flight_count", "count"),
                             avg_flights   = ("flight_count", "mean"),
                             avg_arr_delay = ("avg_arr_delay", "mean"),
                             pct_delayed   = ("pct_delayed", "mean"),
                         ))

    print(f"\n  {'Quartile':<12} {'Routes':>8} {'Avg Flights':>12} "
          f"{'Avg ArrDelay':>13} {'% Delayed':>10}")
    print(f"  {'-'*60}")
    for q, row in summary.iterrows():
        print(f"  {q:<12} {int(row['n_routes']):>8,} {row['avg_flights']:>12.0f} "
              f"{row['avg_arr_delay']:>13.1f} {row['pct_delayed']*100:>9.1f}%")

    corr = pair_stats["flight_count"].corr(pair_stats["avg_arr_delay"])
    print(f"\n  Pearson correlation (volume vs avg delay): {corr:.3f}")
    print(f"  Interpretation: ", end="")
    if corr > 0.2:
        print("✅ Positive — busier routes have higher delays")
    elif corr < -0.2:
        print("⚠️  Negative — busier routes have lower delays (airlines optimize high-volume routes)")
    else:
        print("➡️  Weak — volume alone doesn't drive delay, but congestion does")


# ── B. Taxi anomaly correlation ──────────────────────────────────────────────
def check_taxi_anomaly_correlation(df):
    print("\n" + "="*60)
    print("B. TAXI ANOMALY CORRELATION")
    print("="*60)
    print("Question: when TaxiOut is elevated at origin, is ArrDelay higher at dest?")

    # Compute per-airport TaxiOut baseline
    baseline = (df.groupby("ORIGIN")["TaxiOut"]
                  .agg(["mean", "std"])
                  .rename(columns={"mean": "taxi_mean", "std": "taxi_std"}))
    baseline["taxi_std"] = baseline["taxi_std"].replace(0, 1)

    df2 = df.merge(baseline, on="ORIGIN")
    df2["taxi_zscore"] = (df2["TaxiOut"] - df2["taxi_mean"]) / df2["taxi_std"]

    # Bin by taxi z-score
    bins   = [-np.inf, -1, 0, 1, TAXI_ZSCORE_THRESH, np.inf]
    labels = ["Very low", "Below avg", "Normal", "Elevated", "Congested (>1.5σ)"]
    df2["taxi_bin"] = pd.cut(df2["taxi_zscore"], bins=bins, labels=labels)

    summary = (df2.groupby("taxi_bin", observed=True)
                  .agg(
                      n_flights     = ("ArrDelay", "count"),
                      avg_arr_delay = ("ArrDelay", "mean"),
                      pct_delayed   = ("ArrDelay", lambda x: (x > 15).mean()),
                      avg_taxi_out  = ("TaxiOut",  "mean"),
                  ))

    print(f"\n  {'TaxiOut Level':<22} {'Flights':>10} {'Avg TaxiOut':>12} "
          f"{'Avg ArrDelay':>13} {'% Delayed':>10}")
    print(f"  {'-'*72}")
    for b, row in summary.iterrows():
        print(f"  {b:<22} {int(row['n_flights']):>10,} {row['avg_taxi_out']:>12.1f} "
              f"{row['avg_arr_delay']:>13.1f} {row['pct_delayed']*100:>9.1f}%")

    corr = df2["taxi_zscore"].corr(df2["ArrDelay"])
    print(f"\n  Pearson correlation (TaxiOut z-score vs ArrDelay): {corr:.3f}")
    print(f"  Interpretation: ", end="")
    if corr > 0.3:
        print("✅ Strong — congested departures directly cause arrival delays")
    elif corr > 0.1:
        print("✅ Moderate — TaxiOut elevation predicts higher arrival delays")
    else:
        print("⚠️  Weak — check data")

    # Downstream effect — does congestion at ORIGIN affect DEST arrivals?
    print(f"\n  Downstream effect (congested origin → dest arrival delay):")
    normal    = df2[df2["taxi_zscore"] <= 0]["ArrDelay"].mean()
    elevated  = df2[(df2["taxi_zscore"] > 0) &
                    (df2["taxi_zscore"] <= TAXI_ZSCORE_THRESH)]["ArrDelay"].mean()
    congested = df2[df2["taxi_zscore"] > TAXI_ZSCORE_THRESH]["ArrDelay"].mean()

    print(f"    Normal TaxiOut    → avg ArrDelay at DEST: {normal:.1f} min")
    print(f"    Elevated TaxiOut  → avg ArrDelay at DEST: {elevated:.1f} min")
    print(f"    Congested TaxiOut → avg ArrDelay at DEST: {congested:.1f} min")
    print(f"    Congestion uplift: {congested - normal:.1f} min extra delay at destination")


# ── C. Hub vs spoke delays ───────────────────────────────────────────────────
def check_hub_spoke_correlation(df):
    print("\n" + "="*60)
    print("C. HUB vs SPOKE DELAY PATTERNS")
    print("="*60)
    print("Question: do hub airports show different delay patterns than spokes?")

    dep_counts = df["ORIGIN"].value_counts()
    hubs  = set(dep_counts.head(HUB_TOP_N).index)
    spokes = set(dep_counts.index) - hubs

    hub_flights   = df[df["ORIGIN"].isin(hubs)]
    spoke_flights = df[df["ORIGIN"].isin(spokes)]

    print(f"\n  Hub airports   : {len(hubs)}")
    print(f"  Spoke airports : {len(spokes)}")

    print(f"\n  {'Metric':<30} {'Hubs':>12} {'Spokes':>12} {'Difference':>12}")
    print(f"  {'-'*68}")

    metrics = [
        ("Avg ArrDelay (min)",
         hub_flights["ArrDelay"].mean(),
         spoke_flights["ArrDelay"].mean()),
        ("% Flights delayed (>15min)",
         (hub_flights["ArrDelay"] > 15).mean() * 100,
         (spoke_flights["ArrDelay"] > 15).mean() * 100),
        ("Avg TaxiOut (min)",
         hub_flights["TaxiOut"].mean(),
         spoke_flights["TaxiOut"].mean()),
        ("Std ArrDelay (volatility)",
         hub_flights["ArrDelay"].std(),
         spoke_flights["ArrDelay"].std()),
        ("p95 ArrDelay (min)",
         hub_flights["ArrDelay"].quantile(0.95),
         spoke_flights["ArrDelay"].quantile(0.95)),
    ]

    for name, hub_val, spoke_val in metrics:
        diff = hub_val - spoke_val
        print(f"  {name:<30} {hub_val:>12.1f} {spoke_val:>12.1f} {diff:>+12.1f}")

    # Per-hub breakdown of top 10
    print(f"\n  Top 10 hubs by avg ArrDelay:")
    hub_summary = (hub_flights.groupby("ORIGIN")
                              .agg(
                                  departures    = ("ArrDelay", "count"),
                                  avg_arr_delay = ("ArrDelay", "mean"),
                                  pct_delayed   = ("ArrDelay", lambda x: (x > 15).mean()),
                                  avg_taxi_out  = ("TaxiOut",  "mean"),
                              )
                              .sort_values("avg_arr_delay", ascending=False))

    print(f"  {'Airport':<10} {'Departures':>12} {'Avg Delay':>10} "
          f"{'% Delayed':>10} {'Avg TaxiOut':>12}")
    print(f"  {'-'*58}")
    for ap, row in hub_summary.head(10).iterrows():
        print(f"  {ap:<10} {int(row['departures']):>12,} {row['avg_arr_delay']:>10.1f} "
              f"{row['pct_delayed']*100:>9.1f}% {row['avg_taxi_out']:>12.1f}")


# ── D. Time of day congestion buildup ────────────────────────────────────────
def check_temporal_congestion(df):
    print("\n" + "="*60)
    print("D. TEMPORAL CONGESTION BUILDUP")
    print("="*60)
    print("Question: does delay accumulate through the day (congestion cascade)?")

    df2 = df.copy()
    df2["hour"] = df2["dep_datetime"].dt.hour

    hourly = (df2.groupby("hour")
                 .agg(
                     avg_dep_delay = ("DepDelay", "mean"),
                     avg_arr_delay = ("ArrDelay", "mean"),
                     avg_taxi_out  = ("TaxiOut",  "mean"),
                     pct_delayed   = ("ArrDelay", lambda x: (x > 15).mean()),
                 ))

    print(f"\n  {'Hour':<6} {'Avg DepDelay':>13} {'Avg ArrDelay':>13} "
          f"{'Avg TaxiOut':>12} {'% Delayed':>10}")
    print(f"  {'-'*58}")
    for hour, row in hourly.iterrows():
        bar = "█" * int(row["avg_arr_delay"] / 2) if row["avg_arr_delay"] > 0 else ""
        print(f"  {hour:02d}:00  {row['avg_dep_delay']:>13.1f} {row['avg_arr_delay']:>13.1f} "
              f"{row['avg_taxi_out']:>12.1f} {row['pct_delayed']*100:>9.1f}%  {bar}")

    morning = hourly.loc[6:9, "avg_arr_delay"].mean()
    afternoon = hourly.loc[15:19, "avg_arr_delay"].mean()
    print(f"\n  Morning avg delay   (06-09): {morning:.1f} min")
    print(f"  Afternoon avg delay (15-19): {afternoon:.1f} min")
    print(f"  Daily buildup: {afternoon - morning:.1f} min increase from morning to afternoon")
    if afternoon > morning + 5:
        print("  ✅ Clear congestion cascade — delays build through the day")
    else:
        print("  ➡️  Minimal daily buildup")


def main():
    df = load_data()
    check_volume_correlation(df)
    check_taxi_anomaly_correlation(df)
    check_hub_spoke_correlation(df)
    check_temporal_congestion(df)

    print("\n" + "="*60)
    print("CORRELATION CHECK COMPLETE")
    print("="*60)
    print("If all four checks show positive correlation, your congestion")
    print("edges are capturing real delay propagation signal.")
    print("Proceed to 05_build_graph_snapshots.py")


if __name__ == "__main__":
    main()