import sys
import csv
import datetime
import numpy as np

output_type = np.dtype(
    [
        ("date", "datetime64[s]"),
        ("weather", "b"),
        ("temp", "f")
    ]
)

SUNNY = 0
RAINY = 1
SNOWY = 2
SLEET = 3

last_day_temp = None # used to fill the temp for days which have no temp info

def weather(row: dict):
    water_str = row["Precipitation Water Equiv"]
    water = round(float(water_str), 2) if water_str != "" else 0.0

    snow_str = row["Snowfall"]
    snow = round(float(snow_str), 2) if snow_str != "" else 0.0

    if snow > 0.0 and water > 0:
        return SLEET
    elif water > 0.0:
        return RAINY
    elif snow > 0.0:
        return SNOWY
    else:
        return SUNNY

def parse_date(row: dict):
    dstr = row["Date"]

    d = datetime.datetime.strptime(dstr, '%m/%d/%Y %H:%M:%S')

    return d

def parse_row(row: dict):
    global last_day_temp
    
    date = parse_date(row)
    wh = weather(row)
    temp_str = row["Avg Temp"]

    temp = round(float(temp_str), 2) if temp_str != "" else last_day_temp

    last_day_temp = temp

    return (date, wh, temp)



def process(input_file: str, output_file: str):
    arr: np.ndarray = None
    data: list = None

    with open(input_file, "rt") as fp:
        reader = csv.DictReader(fp)

        data = [parse_row(row) for row in reader]

    print(data[0: 10])

    arr = np.array(data, dtype=output_type)
    np.save(output_file, arr, allow_pickle=False)

def read_to_test(output_file: str):
    arr = np.load(output_file)

    print(arr[0: 10])

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    process(input_file, output_file)

    read_to_test(output_file)