"""Check raw parquet structure to find ArrDelay."""
import pandas as pd
import os

BASE_DIR = r"C:\Users\user\Desktop\Airline_Graphs_Project"

# Check all parquet files in the project
paths = [
    os.path.join(BASE_DIR, "flights_2018_2022.parquet"),
    os.path.join(BASE_DIR, "graph_data", "weather_node_features.parquet"),
    os.path.join(BASE_DIR, "graph_data", "flight_lookup.parquet"),
]

for path in paths:
    if os.path.exists(path):
        df = pd.read_parquet(path, columns=None)
        print(f"\n{os.path.basename(path)}")
        print(f"  Rows    : {len(df):,}")
        print(f"  Columns : {df.columns.tolist()}")
        if "ArrDelay" in df.columns:
            print(f"  ArrDelay sample: {df['ArrDelay'].dropna().head(5).tolist()}")
            print(f"  ArrDelay null%:  {df['ArrDelay'].isna().mean()*100:.1f}%")
    else:
        print(f"\n{os.path.basename(path)}: NOT FOUND")