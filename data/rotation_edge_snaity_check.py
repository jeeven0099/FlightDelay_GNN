import pandas as pd

edges = pd.read_parquet(
    r"C:\Users\user\Desktop\Airline_Graphs_Project\graph_data\rotation_edges.parquet"
)

# Split by whether leg1 was delayed
on_time = edges[edges["leg1_arr_delay"] <= 0]
delayed  = edges[edges["leg1_arr_delay"] > 15]
severe   = edges[edges["leg1_arr_delay"] > 60]

print("Leg2 DepDelay when leg1 was ON TIME:")
print(f"  mean = {on_time['leg2_dep_delay'].mean():.1f} min")

print("Leg2 DepDelay when leg1 was DELAYED (>15 min):")
print(f"  mean = {delayed['leg2_dep_delay'].mean():.1f} min")

print("Leg2 DepDelay when leg1 was SEVERELY DELAYED (>60 min):")
print(f"  mean = {severe['leg2_dep_delay'].mean():.1f} min")

# Also break down by turnaround tightness
print("\nPropagation rate by turnaround window (delayed leg1 only):")
for lo, hi in [(0,30), (30,60), (60,90)]:
    bucket = delayed[
        (delayed["turnaround_min"] >= lo) & 
        (delayed["turnaround_min"] < hi)
    ]
    prop_rate = (bucket["leg2_dep_delay"] > 15).mean()
    print(f"  Turnaround {lo}-{hi} min: "
          f"{prop_rate*100:.1f}% of leg2 flights also delayed "
          f"(n={len(bucket):,})")