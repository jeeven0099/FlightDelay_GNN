import pandas as pd
import numpy as np

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("C:/Users/user/Desktop/Airline_Graphs_Project/data/datasets/bts_flight_data/flights.csv")
df.columns = df.columns.str.strip()
df.columns = df.columns.str.strip()
# =========================
# 2. FIX DATE FORMAT
# =========================
df["FlightDate"] = pd.to_datetime(
    df["FlightDate"],
    format="mixed",
    errors="coerce"
)

# =========================
# 3. REMOVE BAD ROWS
# =========================
df = df[
    (df["Cancelled"] == 0) &
    (df["Diverted"] == 0) &
    (df["Tail_Number"].notna())
].copy()

# =========================
# 4. FIX TIME FUNCTION
# =========================
def convert_hhmm(val):
    if pd.isna(val):
        return np.nan, np.nan
    val = int(val)
    return val // 100, val % 100

# =========================
# 5. BUILD DATETIME COLUMNS
# =========================
def build_datetime(row, col):
    try:
        hour, minute = convert_hhmm(row[col])
        return pd.Timestamp(
            year=row["FlightDate"].year,
            month=row["FlightDate"].month,
            day=row["FlightDate"].day,
            hour=int(hour),
            minute=int(minute)
        )
    except:
        return pd.NaT

df["dep_datetime"] = df.apply(lambda x: build_datetime(x, "DepTime"), axis=1)
df["arr_datetime"] = df.apply(lambda x: build_datetime(x, "ArrTime"), axis=1)

# =========================
# 6. DROP INVALID TIMES
# =========================
df = df[
    df["dep_datetime"].notna() &
    df["arr_datetime"].notna()
]

# =========================
# 7. HANDLE OVERNIGHT FLIGHTS
# =========================
# if arrival < departure → next day
df.loc[df["arr_datetime"] < df["dep_datetime"], "arr_datetime"] += pd.Timedelta(days=1)

# =========================
# 8. CREATE WEATHER ALIGNMENT TIME
# =========================
df["weather_time"] = df["dep_datetime"].dt.floor("H")

# =========================
# 9. SELECT IMPORTANT COLUMNS
# =========================
df_clean = df[[
    "FlightDate",
    "Tail_Number",
    "Operating_Airline",
    "Origin",
    "Dest",
    "dep_datetime",
    "arr_datetime",
    "weather_time",

    "DepDelay",
    "ArrDelay",
    "TaxiOut",
    "TaxiIn",
    "AirTime",
    "Distance"
]].copy()

# =========================
# 10. RENAME FOR CONSISTENCY
# =========================
df_clean.rename(columns={
    "Origin": "ORIGIN",
    "Dest": "DEST"
}, inplace=True)

# =========================
# 11. SORT FOR ROTATION BUILDING
# =========================
df_clean = df_clean.sort_values(
    ["Tail_Number", "dep_datetime"]
).reset_index(drop=True)

# =========================
# 12. SAVE CLEAN DATA
# =========================
df_clean.to_csv("flights_clean.csv", index=False)

print("✅ Cleaning complete!")
print(df_clean.head())