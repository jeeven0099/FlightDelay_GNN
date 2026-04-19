"""Debug what AviationStack actually returns."""
import os
import requests
from dotenv import load_dotenv
load_dotenv()

KEY = os.getenv("AVIATIONSTACK_KEY","")
if not KEY:
    print("No key found in .env")
    exit()

url = "http://api.aviationstack.com/v1/flights"
params = {"access_key":KEY, "dep_iata":"ATL", "limit":2}
r = requests.get(url, params=params, timeout=15)
data = r.json().get("data",[])
if data:
    f = data[0]
    dep = f.get("departure",{})
    arr = f.get("arrival",{})
    air = f.get("aircraft",{})
    print("departure keys:", list(dep.keys()))
    print("departure.scheduled:", dep.get("scheduled"))
    print("departure.actual:",    dep.get("actual"))
    print("arrival keys:",   list(arr.keys()))
    print("arrival.scheduled:", arr.get("scheduled"))
    print("arrival.estimated:", arr.get("estimated"))
    print("aircraft keys:", list(air.keys()))
    print("aircraft.registration:", air.get("registration"))
    print("airline:", f.get("airline",{}).get("iata"))
else:
    print("No data returned")
    print(r.json())