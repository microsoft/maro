# usage :
# export CONFIG=/home/zhanyu/bikeData/maro/examples/citi_bike/q_learning/single_host_mode/config.yml; export PYTHONPATH=/home/zhanyu/bikeData/maro ; ls -dQ ../../../ny/*.csv | xargs -i python station_form_csv.py {} /home/zhanyu/bikeData/ny/full/201306_202001.station.csv /home/zhanyu/bikeData/ny/station/

import numpy as np
import pandas as pd
import json
import os
import io
import sys
import yaml

from maro.utils.dashboard import DashboardBase
from maro.utils import SimpleExperiencePool, Logger, LogFormat, convert_dottable

CONFIG_PATH = os.environ.get('CONFIG') or 'config.yml'

with io.open(CONFIG_PATH, 'r') as in_file:
    raw_config = yaml.safe_load(in_file)
    config = convert_dottable(raw_config)

# Config for dashboard


_dashboard = DashboardBase('city_bike_0318', None,
                           dbname='citi_bike')


def process_bike_data(bike_data_file):
    bike_data = None
    if os.path.exists(bike_data_file):
        with open(bike_data_file, mode="r", encoding="utf-8") as bike_csv_file:
            bike_data = pd.read_csv(bike_csv_file)
            print(bike_data_file)
            bike_data['date'] = pd.to_datetime(bike_data['starttime']).dt.date
    return bike_data


def load_full_station_data(full_station_data_file):
    data = None
    if os.path.exists(full_station_data_file):
        with open(full_station_data_file, mode="r", encoding="utf-8") as full_station_csv_file:
            data = pd.read_csv(full_station_csv_file)

    return data


def _gen_station_data(bike_data):
    gp_station_data_start = bike_data[['start station name', 'start station id', 'start station latitude', 'start station longitude']].drop_duplicates()
    gp_station_data_start.rename(columns={'start station name': 'station_name', 'start station id': 'station_id',
                                          'start station latitude': 'station_latitude', 'start station longitude': 'station_longitude'}, inplace=True)

    gp_station_data_end = bike_data[['end station name', 'end station id', 'end station latitude', 'end station longitude']].drop_duplicates()
    gp_station_data_end.rename(columns={'end station name': 'station_name', 'end station id': 'station_id', 'end station latitude': 'station_latitude', 'end station longitude': 'station_longitude'}, inplace=True)

    station_data = pd.concat([gp_station_data_start, gp_station_data_end]).drop_duplicates().sort_values(by=['station_id'])
    print(station_data)
    return station_data


if __name__ == "__main__":
    #read bike data
    bike_data_file = sys.argv[1]
    full_station_data_file = sys.argv[2]
    tar_folder = sys.argv[3]

    station_data_file = os.path.join(tar_folder, os.path.basename(bike_data_file))

    bike_data = process_bike_data(bike_data_file)

    station_data = _gen_station_data(bike_data)

    full_station_data = load_full_station_data(full_station_data_file)

    with open(station_data_file, mode="w", encoding="utf-8") as station_out_file:
        station_data.to_csv(station_out_file, index=False)

    _dashboard.send(fields={'date': str(bike_data.loc[0, 'date']), 'stations': len(station_data)}, tag={}, measurement='bike_station')

    if full_station_data is None:
        with open(full_station_data_file, mode="w", encoding="utf-8") as full_station_out_file:
            station_data.to_csv(full_station_out_file, index=False)
    else:
        full_station_data = pd.concat([station_data, full_station_data]).drop_duplicates()
        with open(full_station_data_file, mode="w", encoding="utf-8") as full_station_out_file:
            full_station_data.to_csv(full_station_out_file, index=False)
