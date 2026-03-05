import gzip
import shutil
import os
import pandas as pd
input_dir = "noaa_weather_data"

for file in os.listdir(input_dir):
    if file.endswith(".gz"):
        with gzip.open(os.path.join(input_dir, file), "rb") as f_in:
            with open(os.path.join(input_dir, file[:-3]), "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)



