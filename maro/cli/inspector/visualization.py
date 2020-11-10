import argparse
import os

from maro.cli.inspector.cim_dashboard import start_cim_dashboard
from maro.cli.inspector.citi_bike_dashboard import start_citi_bike_dashboard

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("rootpath")
    parser.add_argument("senario")
    args = parser.parse_args()
    ROOT_PATH = args.rootpath
    senario = args.senario

    if senario == 'cim':
        start_cim_dashboard(ROOT_PATH)

    elif senario == 'citi_bike':
        start_citi_bike_dashboard(ROOT_PATH)
