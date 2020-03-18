# usage :
# export CONFIG=/home/zhanyu/bikeData/maro/examples/citi_bike/q_learning/single_host_mode/config.yml; export PYTHONPATH=/home/zhanyu/bikeData/maro ; ls -dQ ../../../ny/*.csv | xargs -i python bike_form_csv.py {} /home/zhanyu/bikeData/ny/full/201306_202001.bike.csv /home/zhanyu/bikeData/ny/bike/

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
            bike_data['date'] = pd.to_datetime(bike_data['starttime']).dt.date
    return bike_data


def load_full_bike_data(full_bike_data_file):
    data = None
    if os.path.exists(full_bike_data_file):
        with open(full_bike_data_file, mode="r", encoding="utf-8") as full_bike_csv_file:
            data = pd.read_csv(full_bike_csv_file)

    return data


def _gp_bike_data(bike_data):
    gp_bike_data = bike_data.groupby(['bikeid']).size()
    gp_bike_data.name = 'rides'
    gp_bike_data = gp_bike_data.reset_index()
    print(gp_bike_data)
    return gp_bike_data


if __name__ == "__main__":
    #read bike data
    bike_data_file = sys.argv[1]
    full_bike_data_file = sys.argv[2]
    tar_folder = sys.argv[3]

    gp_bike_data_file = os.path.join(tar_folder, os.path.basename(bike_data_file))

    bike_data = process_bike_data(bike_data_file)

    gp_bike_data = _gp_bike_data(bike_data)

    full_bike_data = load_full_bike_data(full_bike_data_file)

    with open(gp_bike_data_file, mode="w", encoding="utf-8") as gp_bike_out_file:
        gp_bike_data.to_csv(gp_bike_out_file, index=False)

    _dashboard.send(fields={'date': str(bike_data.loc[0, 'date']), 'bike': len(gp_bike_data), 'rides': gp_bike_data['rides'].sum()}, tag={}, measurement='bike_rides')

    if full_bike_data is None:
        with open(full_bike_data_file, mode="w", encoding="utf-8") as full_bike_out_file:
            gp_bike_data.to_csv(full_bike_out_file, index=False)
    else:
        full_bike_data = pd.concat([gp_bike_data, full_bike_data]).groupby(['bikeid']).sum().reset_index()
        with open(full_bike_data_file, mode="w", encoding="utf-8") as full_bike_out_file:
            full_bike_data.to_csv(full_bike_out_file, index=False)
