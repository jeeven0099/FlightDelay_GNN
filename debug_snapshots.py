"""Quick debug script to inspect test snapshot structure."""
import torch
import pandas as pd

BASE_DIR = r"C:\Users\user\Desktop\Airline_Graphs_Project"
GRAPH_DATA_DIR = BASE_DIR + r"\graph_data"

print("Loading test snapshots...")
snaps = torch.load(GRAPH_DATA_DIR + r"\snapshots_test.pt",
                   map_location="cpu", weights_only=False)
print(f"Total test snapshots: {len(snaps)}")

# Find snapshots for 2022-07-18
target = "2022-07-18"
day_snaps = [s for s in snaps
             if pd.Timestamp(s["airport"].snapshot_time).date()
             == pd.Timestamp(target).date()]
print(f"\nSnapshots for {target}: {len(day_snaps)}")

if not day_snaps:
    # Show what dates ARE available
    all_dates = sorted(set(
        str(pd.Timestamp(s["airport"].snapshot_time).date())
        for s in snaps))
    print(f"\nAvailable dates in test set:")
    print(f"  First 5: {all_dates[:5]}")
    print(f"  Last 5 : {all_dates[-5:]}")
    print(f"  Total days: {len(all_dates)}")
else:
    # Inspect first snapshot of that day
    snap = day_snaps[0]
    print(f"\nFirst snapshot: {snap['airport'].snapshot_time}")
    print(f"  Airport nodes  : {snap['airport'].num_nodes}")
    print(f"  Airport x shape: {snap['airport'].x.shape}")
    print(f"  Airport x dtype: {snap['airport'].x.dtype}")

    # Check flight nodes
    try:
        n_fl = snap["flight"].num_nodes
        print(f"\n  Flight num_nodes: {n_fl}")
        if n_fl > 0:
            print(f"  Flight x shape  : {snap['flight'].x.shape}")
            print(f"  Flight x dtype  : {snap['flight'].x.dtype}")
            print(f"  Has flight_id   : {hasattr(snap['flight'], 'flight_id')}")
            print(f"  Has sched_dep   : {hasattr(snap['flight'], 'scheduled_dep')}")
            print(f"  Has sched_arr   : {hasattr(snap['flight'], 'scheduled_arr')}")
        else:
            print("  ⚠ No flight nodes in this snapshot")
    except Exception as e:
        print(f"  ⚠ Flight node error: {e}")

    # Check all snapshots for that day
    fl_counts = []
    for s in day_snaps:
        try:
            fl_counts.append(s["flight"].num_nodes)
        except:
            fl_counts.append(0)
    print(f"\n  Flight node counts across 24h:")
    print(f"    Min: {min(fl_counts)}  Max: {max(fl_counts)}  "
          f"Mean: {sum(fl_counts)/len(fl_counts):.0f}")
    print(f"    Snapshots with 0 flights: "
          f"{sum(1 for c in fl_counts if c == 0)}")
    print(f"    Counts by hour: {fl_counts}")

    # Check edge types
    print(f"\n  Edge types in first snapshot:")
    for et in [("airport","rotation","airport"),
               ("flight","departs_from","airport"),
               ("flight","arrives_at","airport")]:
        try:
            n = snap[et].edge_index.shape[1]
            print(f"    {str(et):<45} {n} edges")
        except:
            print(f"    {str(et):<45} NOT FOUND")