import csv
import json
import os

import numpy as np
import pandas as pd
import tqdm
import yaml

from maro.cli.inspector.launch_env_dashboard import launch_dashboard
from maro.cli.utils.params import GlobalPaths
from maro.utils.logger import CliLogger

logger = CliLogger(name=__name__)

NAME_CONVERSION_PATH = GlobalPaths.MARO_INSPECTOR_FILE_PATH["name_conversion_path"]
PORTS_FILE_PATH = GlobalPaths.MARO_INSPECTOR_FILE_PATH["ports_file_path"]
VESSELS_FILE_PATH = GlobalPaths.MARO_INSPECTOR_FILE_PATH["vessels_file_path"]
STATIONS_FILE_PATH = GlobalPaths.MARO_INSPECTOR_FILE_PATH["stations_file_path"]


def init_csv(file_path, header):
    """Clean & init summary csv file.

    Args:
        File_path(str): summary file path.
        Header(list): expected header of summary file.
    """
    if os.path.exists(file_path):
        os.remove(file_path)
    with open(file_path, "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()


def summary_append(scenario, dir_epoch, file_name, header, sum_dataframe, i, output_path):
    """Calculate summary info and generate corresponding csv file.
    To accelerate, change each column into numpy.array.

    Args:
        scenario(str): Current scenario.
        This parameter is useless right now. Cause only scenario-cim needs operations within this function.
        dir_epoch(str): Current epoch.
        Loop is outside this function. This function calculate a summary within an epoch each time.
        file_name(str): Name of file needed to be summarized.
        Some scenario has multiple files within one epoch. e.g.cim.
        header(list): List of columns needed to be summarized.
        sum_dataframe(Dataframe): Temporary dataframe to restore results.
        i(int): Index (to indicate the progress in loop).
        output_path(str): Path of output CSV file.

    """
    input_path = os.path.join(dir_epoch, file_name)
    data = pd.read_csv(input_path)
    data_insert = []
    for ele in header:
        data_insert.append(np.sum(np.array(data[ele]), axis=0))
    sum_dataframe.loc[i] = data_insert
    sum_dataframe.to_csv(output_path, header=True, index=True)


def generate_summary(scenario, ROOT_PATH):
    """Generate summary info of current scenario.
    Different scenario has different data features.
    e.g. cim has multiple epochs while citi_bike only has one.
    Each scenario should be treated respectively.

    Args:
        scenario(str): Current scenario.
        ROOT_PATH(str): Data folder path.

    """
    ports_header = ["capacity", "empty", "full", "on_shipper", "on_consignee", "shortage", "booking", "fulfillment"]
    # vessels_header = ["capacity", "empty", "full", "remaining_space", "early_discharge"]
    stations_header = ["bikes", "shortage", "trip_requirement", "fulfillment", "capacity"]
    dbtype_list_all = os.listdir(ROOT_PATH)
    temp_len = len(dbtype_list_all)
    dbtype_list = []
    for index in range(0, temp_len):
        if os.path.exists(os.path.join(ROOT_PATH, r"snapshot_" + str(index))):
            dbtype_list.append(os.path.join(ROOT_PATH, r"snapshot_" + str(index)))

    if scenario == "cim":
        init_csv(os.path.join(ROOT_PATH, PORTS_FILE_PATH), ports_header)
        # init_csv(vessels_file_path, vessels_header)
        ports_sum_dataframe = pd.read_csv(os.path.join(ROOT_PATH, PORTS_FILE_PATH),
                                        names=ports_header)
        # vessels_sum_dataframe = pd.read_csv(vessels_file_path, names=vessels_header)
    else:
        init_csv(os.path.join(ROOT_PATH, STATIONS_FILE_PATH), stations_header)
    if scenario == "cim":
        i = 1
        for i in tqdm.tqdm(range(len(dbtype_list))):
            dbtype = dbtype_list[i]
            dir_epoch = os.path.join(ROOT_PATH, dbtype)
            if not os.path.isdir(dir_epoch):
                continue
            summary_append(scenario, dir_epoch, "ports.csv", ports_header, ports_sum_dataframe,
                           i, os.path.join(ROOT_PATH, PORTS_FILE_PATH))
            # summary_append(dir_epoch, "vessels.csv", vessels_header, vessels_sum_dataframe, i,vessels_file_path)
            i = i + 1
    elif scenario == "CITI_BIKE":
        data = pd.read_csv(os.path.join(ROOT_PATH, "snapshot_0", "stations.csv"))
        data = data[["bikes", "trip_requirement", "fulfillment", "capacity"]].groupby(data["name"]).sum()
        data["fulfillment_ratio"] = list(
            map(lambda x, y: float("{:.4f}".format(x / (y + 1 / 1000))), data["fulfillment"],
                data["trip_requirement"]))
        data.to_csv(os.path.join(ROOT_PATH, STATIONS_FILE_PATH))


def get_holder_name_conversion(scenario, ROOT_PATH, CONVER_PATH):
    """ Generate a CSV File which indicates the relationship between index and holder"s name.

    Args:
        scenario(str): Current scenario. Different scenario has different type of mapping file.
        ROOT_PATH(str): Data folder path.
        CONVER_PATH(str): Path of origin mapping file.

    """
    CONVER_PATH = os.path.join(ROOT_PATH, CONVER_PATH)
    if os.path.exists(os.path.join(ROOT_PATH, NAME_CONVERSION_PATH)):
        os.remove(os.path.join(ROOT_PATH, NAME_CONVERSION_PATH))
    if scenario == "citi_bike":
        with open(CONVER_PATH, "r", encoding="utf8")as fp:
            json_data = json.load(fp)
            name_list = []
            for item in json_data["data"]["stations"]:
                name_list.append(item["name"])
            df = pd.DataFrame(name_list)
            df.to_csv(os.path.join(ROOT_PATH, NAME_CONVERSION_PATH), index=False)
    elif scenario == "cim":
        f = open(CONVER_PATH, "r")
        ystr = f.read()
        aa = yaml.load(ystr, Loader=yaml.FullLoader)
        key_list = aa["ports"].keys()
        df = pd.DataFrame(list(key_list))
        df.to_csv(os.path.join(ROOT_PATH, NAME_CONVERSION_PATH), index=False)


def start_vis(input: str, force: str, **kwargs):
    try:
        import altair
        import streamlit
    except ImportError:
        os.system("pip install streamlit altair")
    ROOT_PATH = input
    FORCE = force
    if not os.path.exists(ROOT_PATH):
        logger.warning_yellow("input path not exists")
        os._exit(0)
    # path to restore summary files
    if FORCE == "yes":
        logger.info("Dashboard Data Processing")
        manifest_file = open(os.path.join(ROOT_PATH, "snapshot.manifest"), "r")
        props_origin = manifest_file.read()
        props = yaml.load(props_origin, Loader=yaml.FullLoader).split()
        scenario = props[0][9:]
        CONVER_PATH = props[1][9:]

        logger.info_green("[1/2]:Generate Name Conversion File.")
        get_holder_name_conversion(scenario, ROOT_PATH, CONVER_PATH)
        logger.info_green("[1/2]:Generate Name Conversion File Done.")

        logger.info_green("[2/2]:Generate Summary.")
        generate_summary(scenario, ROOT_PATH)
        logger.info_green("[2/2]:Generate Summary Done.")
    elif FORCE == "no":
        logger.info_green("Skip Data Generation")
    launch_dashboard(ROOT_PATH, scenario)
