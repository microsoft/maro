# usage:
# python bikedp.py ../../ny ../../ny/bin ../../ny/full/h3_201306_202001.station.csv

import os
import sys
import csv
import math
import datetime
import numpy as np
import pandas as pd

# from geopy import distance
from collections import defaultdict

######### options ##########
"""
"2013-07 - Citi Bike trip data.csv",
"2013-08 - Citi Bike trip data.csv",  
"2013-09 - Citi Bike trip data.csv",  
"2013-10 - Citi Bike trip data.csv",  
"2013-11 - Citi Bike trip data.csv",  
"2013-12 - Citi Bike trip data.csv",  
"2014-01 - Citi Bike trip data.csv",
"2014-02 - Citi Bike trip data.csv",
"""
######### full data ##########
"""
"201306-citibike-tripdata.csv",
"201307-citibike-tripdata.csv",
"201308-citibike-tripdata.csv",
"201309-citibike-tripdata.csv",
"201310-citibike-tripdata.csv",
"201311-citibike-tripdata.csv",
"201312-citibike-tripdata.csv",
"201401-citibike-tripdata.csv",
"201402-citibike-tripdata.csv",
"201403-citibike-tripdata.csv",
"201404-citibike-tripdata.csv",
"201405-citibike-tripdata.csv",
"201406-citibike-tripdata.csv",
"201407-citibike-tripdata.csv",
"201408-citibike-tripdata.csv",
"201409-citibike-tripdata.csv",
"201410-citibike-tripdata.csv",
"201411-citibike-tripdata.csv",
"201412-citibike-tripdata.csv",
"201501-citibike-tripdata.csv",
"201502-citibike-tripdata.csv",
"201503-citibike-tripdata.csv",
"201504-citibike-tripdata.csv",
"201505-citibike-tripdata.csv",
"201506-citibike-tripdata.csv",
"201507-citibike-tripdata.csv",
"201508-citibike-tripdata.csv",
"201509-citibike-tripdata.csv",
"201510-citibike-tripdata.csv",
"201511-citibike-tripdata.csv",
"201512-citibike-tripdata.csv",
"201601-citibike-tripdata.csv",
"201602-citibike-tripdata.csv",
"201603-citibike-tripdata.csv",
"201604-citibike-tripdata.csv",
"201605-citibike-tripdata.csv",
"201606-citibike-tripdata.csv",
"201607-citibike-tripdata.csv",
"201608-citibike-tripdata.csv",
"201609-citibike-tripdata.csv",
"201610-citibike-tripdata.csv",
"201611-citibike-tripdata.csv",
"201612-citibike-tripdata.csv",
"201701-citibike-tripdata.csv",
"201702-citibike-tripdata.csv",
"201703-citibike-tripdata.csv",
"201704-citibike-tripdata.csv",
"201705-citibike-tripdata.csv",
"201706-citibike-tripdata.csv",
"201707-citibike-tripdata.csv",
"201708-citibike-tripdata.csv",
"201709-citibike-tripdata.csv",
"201710-citibike-tripdata.csv",
"201711-citibike-tripdata.csv",
"201712-citibike-tripdata.csv",
"201801-citibike-tripdata.csv",
"201802-citibike-tripdata.csv",
"201803-citibike-tripdata.csv",
"201804-citibike-tripdata.csv",
"201805-citibike-tripdata.csv",
"201806-citibike-tripdata.csv",
"201807-citibike-tripdata.csv",
"201808-citibike-tripdata.csv",
"201809-citibike-tripdata.csv",
"201810-citibike-tripdata.csv",
"201811-citibike-tripdata.csv",
"201812-citibike-tripdata.csv",
"201901-citibike-tripdata.csv",
"201902-citibike-tripdata.csv",
"201903-citibike-tripdata.csv",
"201904-citibike-tripdata.csv",
"201905-citibike-tripdata.csv",
"201906-citibike-tripdata.csv",
"201907-citibike-tripdata.csv",
"201908-citibike-tripdata.csv",
"201909-citibike-tripdata.csv",
"201910-citibike-tripdata.csv",
"201911-citibike-tripdata.csv",
"201912-citibike-tripdata.csv",
"202001-citibike-tripdata.csv"
"""
# NOTE: the order will affect the result data
input_file_list = [
    "201912-citibike-tripdata.csv"
]

usertype_map={
    "Subscriber": 0,
    "Customer": 1
}

data_file_name = "data.bin"
mapping_file_name = "map.csv"
distance_table_name = "distance.csv"


output_data_dtype = np.dtype([
    ("start_time", "datetime64[s]"), # datetime
    ("start_station", "i4"), # id
    ("end_station", "i4"), # id
    ("duration", "i4"), # min
    ("gendor", "b"), 
    ("usertype", "b"), 
])


######### input ############

def deal_usertype(dt: str):
    d = dt.strip("\"")

    #
    return usertype_map[d] if d in usertype_map else 2

def deal_datetime(dt: str):
    try:
        d = datetime.datetime.strptime(dt, '%Y-%m-%d %H:%M:%S.%f')
    except ValueError:
        d = datetime.datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')

    return d

def deal_str(d: str):
    return d.strip("\"")

def deal_int(d: str):
    return int(d.strip("\""))

def deal_float(d: str):
    return float(d.strip("\|"))

def cal_durations(durations: int):
    # tranform duration into min
    return math.ceil(durations/60)

def read_src_file(file: str):
    """read and return processed rows"""
    ret = []
    stations = {}

    if os.path.exists(file):
        with open(file) as fp:

            reader = csv.DictReader(fp)

            for l in reader:
                item = (
                    cal_durations(deal_int(l["tripduration"])),
                    deal_datetime(l["starttime"]),
                    deal_int(l["start station id"]),
                    deal_str(l["start station name"]),
                    deal_int(l["end station id"]),
                    deal_str(l['end station name']),
                    deal_float(l["start station latitude"]),
                    deal_float(l["start station longitude"]),
                    deal_float(l["end station latitude"]),
                    deal_float(l["end station longitude"]),
                    deal_usertype(l["usertype"]),
                    deal_int(l["gender"])
                )

                ret.append(item)

                # stations[item[2]] = (item[3], item[6], item[7]) # start station, log and lat
                # stations[item[4]] = (item[5], item[8], item[9]) # end station, log and lat

    return ret, stations

def distinct_stations(s: dict, d: dict):
    for k, v in d.items():
        s[k] = v

    return s


######### output ############


def init(output_folder: str):
    data_path = os.path.join(output_folder, data_file_name)
    map_path = os.path.join(output_folder, mapping_file_name)

    np.memmap()


def concat(data: list, file: str, station_data: pd.DataFrame):
    ret = []

    item_num = len(data)

    for d in data:
        from_cell_id = station_data.loc[int(station_data['station_id']) == d[2], 'hex_id']
        to_cell_id = station_data.loc[int(station_data['station_id']) == d[4], 'hex_id']
        print((
            d[1],
            d[2],
            d[4],
            d[0],
            d[10],
            d[11],
            from_cell_id,
            to_cell_id
        ))
        ret.append((
            d[1],
            d[2],
            d[4],
            d[0],
            d[10],
            d[11],
            from_cell_id,
            to_cell_id
        ))

    # get the file size
    file_size = 0

    if not os.path.exists(file):
        with open(file, "w+") as fp:
            pass

    with open(file, "rb") as fp:
        fp.seek(0,2)
        file_size = fp.tell()

    # append to the end
    arr = np.memmap(file, dtype=output_data_dtype, offset=file_size, shape=(item_num, ))

    arr[:] = np.array(ret, dtype=output_data_dtype)


def save_mapping(data: dict, file: str):

    # id->name mapping
    with open(file, "w+") as fp:
        for k, v in data.items():
            fp.write(f"{k},{v[0]}\n")

def sort_by_distance(station):
    return station[1]

def save_distance_table(data: dict, file: str):
    # distance table
    # each row compose with station id and sorted distance to each other station, like:
    # statation_id, (station_2, distance), (station_4, distance)

    with open(file, "w+") as fp:
        writer = csv.writer(fp)

        for id1, s1 in data.items():
            dist_list = []

            for id2, s2 in data.items():
                if id1 != id2:
                    dist_list.append((id2, round(distance.distance((s1[1], s1[2]), (s2[1], s2[2])).m, 4)))

            dist_list.sort(key=sort_by_distance)

            row_data = [id1]

            for id, dist in dist_list:
                row_data.append(id )
                row_data.append(dist)

            writer.writerow(
                row_data
            )

def read_station_file(station_file_path):
    h3_data = None
    if os.path.exists(station_file_path):
        with open(station_file_path, mode="r", encoding="utf-8") as station_file:
            h3_data = pd.read_csv(station_file)
    return h3_data


if __name__ == "__main__":
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    station_file_path = sys.argv[3]

    output_data_path = os.path.join(output_folder, data_file_name)

    if len(sys.argv) >= 5:
        # tick type
        time_lvl = sys.argv[4]
    
    if len(sys.argv) >= 6:
        duration_type = sys.argv[5]

    station_map = {}

    station_data = read_station_file(station_file_path)

    for src_file in input_file_list:
        src_full_path = os.path.join(input_folder, src_file)

        r,s = read_src_file(src_full_path)

        # s = distinct_stations(station_map, s)

        concat(r, output_data_path, station_data)

    # map_full_path = os.path.join(output_folder, mapping_file_name)
    # save_mapping(station_map, map_full_path)

    # distance_full_path = os.path.join(output_folder, distance_table_name)
    # save_distance_table(station_map, distance_full_path)