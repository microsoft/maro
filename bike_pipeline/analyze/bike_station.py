# usage
# export CONFIG=/home/zhanyu/bikeData/maro/examples/citi_bike/q_learning/single_host_mode/config.yml; export PYTHONPATH=/home/zhanyu/bikeData/maro ; ls -dQ ../../../ny/station/*.csv | xargs -i python bike_station.py {} /home/zhanyu/bikeData/maro/bike_pipeline/data/sample_station.json /home/zhanyu/bikeData/ny/station/init


import numpy as np
import pandas as pd
import json
import os
import io
import sys
import yaml

from maro.utils.dashboard import DashboardBase

_dashboard = DashboardBase('city_bike_0321', None,
                           dbname='citi_bike')


def load_csv_station_data(station_csv_file):
    station_data = None
    if os.path.exists(station_csv_file):
        with open(station_csv_file, mode="r", encoding="utf-8") as station_csv_file:
            station_data = pd.read_csv(station_csv_file)

    return station_data


def process_station_data(station_data_file):
    station_data = None
    if os.path.exists(station_data_file):
        with open(station_data_file, mode="r", encoding="utf-8") as station_json_file:
            raw_station_data = pd.read_json(station_json_file)['features']
            station_data = raw_station_data.apply(_station_json_to_pd)
            print(station_data)

            print("bikes: ", station_data['bikes'].sum())
            print("capacity: ", station_data['capacity'].sum())

    return station_data


def _station_json_to_pd(json_data):
    json_frame = pd.Series(
        [json_data['geometry']['coordinates'][0],
         json_data['geometry']['coordinates'][1],
         json_data['properties']['station']['id'],
         json_data['properties']['station']['name'],
         json_data['properties']['station']['capacity'],
         json_data['properties']['station']['bikes_available']
         ],
        index=['station_longitude', 'station_latitude', 'station_id', 'station_name', 'capacity', 'bikes'])
    json_frame['station_id'] = pd.to_numeric(json_frame['station_id'],errors='coerce',downcast='integer')
    json_frame['station_latitude'] = pd.to_numeric(json_frame['station_latitude'],errors='coerce',downcast='integer')
    json_frame['station_longitude'] = pd.to_numeric(json_frame['station_longitude'],errors='coerce',downcast='integer')


if __name__ == "__main__":
    #read bike data
    station_csv_file = sys.argv[1]
    station_json_file = sys.argv[2]
    tar_folder = sys.argv[3]

    station_data = load_csv_station_data(station_csv_file)

    sample_station_data = process_station_data(station_json_file)

    compare_station = pd.concat([station_data[['station_id', 'station_name', 'station_longitude', 'station_latitude']], 
        sample_station_data[['station_id', 'station_name', 'station_longitude', 'station_latitude']]]).drop_duplicates(subset=['station_id']).sort_values(by=['station_id'])
    compare_station['in_csv'] = compare_station['station_id'].isin(station_data['station_id'])
    compare_station['in_json'] = compare_station['station_id'].isin(sample_station_data['station_id'])
    compare_station = compare_station.join(sample_station_data[['station_id', 'capacity', 'bikes']].set_index('station_id'), on='station_id')
    mean_capacity = np.floor(sample_station_data['capacity'].mean())
    mean_bike = np.floor(sample_station_data['bikes'].mean())
    init_bike_rate = mean_bike/mean_capacity
    compare_station.loc[compare_station['in_json'] == False, 'capacity'] = mean_capacity
    compare_station.loc[compare_station['in_json'] == False, 'bikes'] = mean_bike
    compare_station['init'] = (compare_station['capacity'] * init_bike_rate).apply(np.floor)

    print(compare_station)
    station_init_file = os.path.join(tar_folder, os.path.basename(station_csv_file))
    print(station_init_file)
    with open(station_init_file, mode="w", encoding="utf-8") as station_out_file:
        compare_station.to_csv(station_out_file, index=False)
