import argparse

from maro.cli.inspector.cim_dashboard import start_cim_dashboard
from maro.cli.inspector.citi_bike_dashboard import start_citi_bike_dashboard
from maro.cli.utils.params import GlobalScenarios

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("rootpath")
    parser.add_argument("scenario")
    parser.add_argument("epoch_num")
    args = parser.parse_args()

    ROOT_PATH = args.rootpath
    scenario = GlobalScenarios(int(args.scenario))
    epoch_num = int(args.epoch_num)

    if scenario == GlobalScenarios.CIM:
        start_cim_dashboard(ROOT_PATH, epoch_num)

    elif scenario == GlobalScenarios.CITI_BIKE:
        start_citi_bike_dashboard(ROOT_PATH)
