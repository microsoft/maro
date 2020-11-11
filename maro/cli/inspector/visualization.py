import argparse

from maro.cli.inspector.cim_dashboard import start_cim_dashboard
from maro.cli.inspector.citi_bike_dashboard import start_citi_bike_dashboard
from maro.cli.utils.params import GlobalScenaios

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("rootpath")
    parser.add_argument("scenario")
    args = parser.parse_args()
    ROOT_PATH = args.rootpath
    scenario = GlobalScenaios(int(args.scenario))

    if scenario == GlobalScenaios.cim:
        start_cim_dashboard(ROOT_PATH)

    elif scenario == GlobalScenaios.citi_bike:
        start_citi_bike_dashboard(ROOT_PATH)
