import csv
import json
import os

import numpy as np
import pandas as pd
import tqdm
import yaml

from maro.cli.inspector.launch_env_dashboard import launch_dashboard
from maro.cli.utils.params import GlobalFilePaths as Gfiles
from maro.cli.utils.params import GlobalScenarios
from maro.utils.exception.cli_exception import CliException
from maro.utils.logger import CliLogger

logger = CliLogger(name=__name__)


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


def generate_summary(scenario, root_path):
    """Generate summary info of current scenario.
    Different scenario has different data features.
    e.g. cim has multiple epochs while citi_bike only has one.
    Each scenario should be treated respectively.

    Args:
        scenario(str): Current scenario.
        root_path(str): Data folder path.

    """
    ports_header = ["capacity", "empty", "full", "on_shipper", "on_consignee", "shortage", "booking", "fulfillment"]
    # vessels_header = ["capacity", "empty", "full", "remaining_space", "early_discharge"]
    stations_header = ["bikes", "shortage", "trip_requirement", "fulfillment", "capacity"]
    dbtype_list_all = os.listdir(root_path)
    temp_len = len(dbtype_list_all)
    dbtype_list = []
    for index in range(0, temp_len):
        if os.path.exists(os.path.join(root_path, f"snapshot_{index}")):
            dbtype_list.append(os.path.join(root_path, f"snapshot_{index}"))

    if scenario == GlobalScenarios.CIM:
        init_csv(os.path.join(root_path, Gfiles.ports_sum), ports_header)
        # init_csv(vessels_file_path, vessels_header)
        ports_sum_dataframe =\
            pd.read_csv(os.path.join(root_path, Gfiles.ports_sum), names=ports_header)
        # vessels_sum_dataframe = pd.read_csv(vessels_file_path, names=vessels_header)
    else:
        init_csv(os.path.join(root_path, Gfiles.stations_sum), stations_header)
    if scenario == GlobalScenarios.CIM:
        i = 1
        for i in tqdm.tqdm(range(len(dbtype_list))):
            dbtype = dbtype_list[i]
            dir_epoch = os.path.join(root_path, dbtype)
            if not os.path.isdir(dir_epoch):
                continue
            summary_append(
                scenario, dir_epoch, "ports.csv", ports_header,
                ports_sum_dataframe, i, os.path.join(root_path, Gfiles.ports_sum))
            # summary_append(dir_epoch, "vessels.csv", vessels_header, vessels_sum_dataframe, i,vessels_file_path)
            i = i + 1
    elif scenario == GlobalScenarios.CITI_BIKE:
        data = pd.read_csv(os.path.join(root_path, "snapshot_0", "stations.csv"))
        data = data[["bikes", "trip_requirement", "fulfillment", "capacity"]].groupby(data["name"]).sum()
        data["fulfillment_ratio"] = list(
            map(lambda x, y: float("{:.4f}".format(x / (y + 1 / 1000))), data["fulfillment"],
                data["trip_requirement"]))
        data.to_csv(os.path.join(root_path, Gfiles.stations_sum))


def get_holder_name_conversion(scenario, root_path, conver_path):
    """ Generate a CSV File which indicates the relationship between index and holder"s name.

    Args:
        scenario(str): Current scenario. Different scenario has different type of mapping file.
        root_path(str): Data folder path.
        conver_path(str): Path of origin mapping file.

    """
    conver_path = os.path.join(root_path, conver_path)
    if os.path.exists(os.path.join(root_path, Gfiles.name_convert)):
        os.remove(os.path.join(root_path, Gfiles.name_convert))
    if scenario == GlobalScenarios.CITI_BIKE:
        with open(conver_path, "r", encoding="utf8")as fp:
            json_data = json.load(fp)
            name_list = []
            for item in json_data["data"]["stations"]:
                name_list.append(item["name"])
            df = pd.DataFrame({"name": name_list})
            df.to_csv(os.path.join(root_path, Gfiles.name_convert), index=False)
    elif scenario == GlobalScenarios.CIM:
        f = open(conver_path, "r")
        ystr = f.read()
        aa = yaml.load(ystr, Loader=yaml.FullLoader)
        key_list = aa["ports"].keys()
        df = pd.DataFrame(list(key_list))
        df.to_csv(os.path.join(root_path, Gfiles.name_convert), index=False)


def start_vis(source: str, force: str, **kwargs):
    """Entrance of data pre-processing.
    Generate name_conversion CSV file & summary file.

    Expected File Structure:
    -input_file_folder_path
        --snapshot_0 : data of each epoch
            --holder_info.csv: Attributes of current epoch
        ………………
        --snapshot_{epoch_num-1}
        --snapshot.manifest: record basic info like scenario name, name of index_name_mapping file
        --index_name_mapping file: record the relationship between an index and its name.
        type of this file varied between scenario.

        summary file would be generated after data processing.

    Args:
        input(str): Data folder path.
        force(str): expected input is yes/no. Indicates whether regenerate data.
        **kwargs:

    """
    root_path = source
    FORCE = force
    if not os.path.exists(root_path):
        raise CliException("input path is not correct. ")
        os._exit(0)
    if not os.path.exists(os.path.join(root_path, "snapshot_0")):
        raise CliException("No data under input folder path. ")
        os._exit(0)
    if not os.path.exists(os.path.join(root_path, "manifest.yml")):
        raise CliException("Manifest file missed. ")
        os._exit(0)

    manifest_file = open(os.path.join(root_path, "manifest.yml"), "r")
    props_origin = manifest_file.read()
    props = yaml.load(props_origin, Loader=yaml.FullLoader)
    scenario = GlobalScenarios[str(props["scenario"]).upper()]
    conver_path = str(props["mappings"])
    epoch_num = int(props["epoch_num"])

    if FORCE == "true":
        logger.info("Dashboard Data Processing")

        get_holder_name_conversion(scenario, root_path, conver_path)
        logger.info_green("[1/2]:Generate Name Conversion File Done.")

        logger.info_green("[2/2]:Generate Summary.")
        generate_summary(scenario, root_path)
        logger.info_green("[2/2]:Generate Summary Done.")
    elif FORCE == "false":
        logger.info_green("Skip Data Generation")
        if not os.path.exists(os.path.join(root_path, Gfiles.name_convert)):
            raise CliException("Have to regenerate data. Name Conversion File is missed. ")
            os._exit(0)
        if scenario == GlobalScenarios.CIM:
            if not os.path.exists(os.path.join(root_path, Gfiles.ports_sum)):
                raise CliException("Have to regenerate data. Summary File is missed. ")
                os._exit(0)
        if scenario == GlobalScenarios.CITI_BIKE:
            if not os.path.exists(os.path.join(root_path, Gfiles.stations_sum)):
                raise CliException("Have to regenerate data. Summary File is missed. ")
                os._exit(0)

    launch_dashboard(root_path, scenario, epoch_num)
