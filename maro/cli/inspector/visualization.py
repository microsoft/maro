# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse

from maro.cli.inspector.cim_dashboard import start_cim_dashboard
from maro.cli.inspector.citi_bike_dashboard import start_citi_bike_dashboard
from maro.cli.inspector.params import GlobalScenarios

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=str)
    parser.add_argument("--scenario", type=str)
    parser.add_argument("--epoch_num", type=int)
    parser.add_argument("--prefix", type=str)
    args = parser.parse_args()

    source_path = args.source_path
    scenario = GlobalScenarios(args.scenario)
    epoch_num = args.epoch_num
    prefix = args.prefix

    if scenario == GlobalScenarios.CIM:
        start_cim_dashboard(source_path, epoch_num, prefix)

    elif scenario == GlobalScenarios.CITI_BIKE:
        start_citi_bike_dashboard(source_path, epoch_num, prefix)
