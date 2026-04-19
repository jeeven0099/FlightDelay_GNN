"""Debug flight_lookup date filtering."""
import pandas as pd

BASE_DIR = r"C:\Users\user\Desktop\Airline_Graphs_Project"
lookup_path = BASE_DIR + r"\graph_data\flight_lookup.parquet"

print("Loading flight_lookup ...")
fl = pd.read_parquet(lookup_path)
print(f"Rows: {len(fl):,}")
print(f"Columns: {fl.columns.tolist()}")
print(f"\ndep_datetime dtype: {fl['dep_datetime'].dtype}")
print(f"Sample values:")
print(fl["dep_datetime"].head(3).tolist())

# Try conversion
fl["dep_datetime"] = pd.to_datetime(fl["dep_datetime"], errors="coerce")
print(f"\nAfter to_datetime dtype: {fl['dep_datetime'].dtype}")
print(f"NaT count: {fl['dep_datetime'].isna().sum():,}")
print(f"Min date: {fl['dep_datetime'].min()}")
print(f"Max date: {fl['dep_datetime'].max()}")

# Check 2022 data
fl_2022 = fl[fl["dep_datetime"].dt.year == 2022]
print(f"\n2022 flights: {len(fl_2022):,}")

if len(fl_2022) > 0:
    dates_2022 = sorted(fl_2022["dep_datetime"].dt.date.unique())
    print(f"First 5 dates in 2022: {dates_2022[:5]}")
    print(f"Last 5 dates in 2022:  {dates_2022[-5:]}")

    # Try the specific date
    target = pd.Timestamp("2022-07-18").date()
    count = (fl_2022["dep_datetime"].dt.date == target).sum()
    print(f"\nFlights on 2022-07-18: {count:,}")
else:
    print("No 2022 data found!")
    print("Checking what years are present:")
    print(fl["dep_datetime"].dt.year.value_counts().sort_index())