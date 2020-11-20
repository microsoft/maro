import csv
import json
import os

import numpy as np
import pandas as pd
import tqdm
import yaml

from maro.utils.exception.cli_exception import CliException
from maro.utils.logger import CliLogger

from .launch_env_dashboard import launch_dashboard
from .params import GlobalFilePaths, GlobalScenarios

logger = CliLogger(name=__name__)


def start_vis(source_path: str, force: str, **kwargs):
    """Entrance of data pre-processing.

        Generate name_conversion CSV file and summary file.

    Expected File Structure:
    -input_file_folder_path
        --epoch_0 : data of each epoch
            --holder_info.csv: Attributes of current epoch
        ………………
        --epoch_{epoch_num-1}
        --manifest.yml: record basic info like scenario name, name of index_name_mapping file
        --index_name_mapping file: record the relationship between an index and its name.
        type of this file varied between scenario.

        summary file would be generated after data processing.

    Args:
        source_path(str): Data folder path.
        force(str): expected input is True/False. Indicates whether regenerate data.
        **kwargs:

    """
    if not os.path.exists(os.path.join(source_path, "manifest.yml")):
        raise CliException("Manifest file missed. ")
        os._exit(0)
    manifest_file = open(os.path.join(source_path, "manifest.yml"), "r")
    manifest_file_content = manifest_file.read()
    settings = yaml.load(manifest_file_content, Loader=yaml.FullLoader)
    scenario = GlobalScenarios[str(settings["scenario"]).upper()]
    conversion_path = str(settings["mappings"])
    epoch_num = int(settings["dump_details"]["epoch_num"])
    prefix = settings["dump_details"]["prefix"]

    if not os.path.exists(source_path):
        raise CliException("input path is not correct. ")
    elif not os.path.exists(os.path.join(source_path, f"{prefix}0")):
        raise CliException("No data under input folder path. ")

    if force:
        logger.info("Dashboard Data Generating")
        _get_holder_name_conversion(scenario, source_path, conversion_path)
        logger.info_green("[1/2]:Generate Name Conversion File Done.")
        logger.info_green("[2/2]:Generating Summary.")
        _generate_summary(scenario, source_path, prefix, epoch_num)
        logger.info_green("[2/2]:Generate Summary Done.")

    else:
        logger.info_green("Skip Data Generation")
        if not os.path.exists(os.path.join(source_path, GlobalFilePaths.name_convert)):
            raise CliException("Have to regenerate data. Name Conversion File is missed.")

        if scenario == GlobalScenarios.CIM:
            if not os.path.exists(os.path.join(source_path, GlobalFilePaths.ports_sum)):
                raise CliException("Have to regenerate data. Summary File is missed.")
        elif scenario == GlobalScenarios.CITI_BIKE:
            if not os.path.exists(os.path.join(source_path, GlobalFilePaths.stations_sum)):
                raise CliException("Have to regenerate data. Summary File is missed.")

    launch_dashboard(source_path, scenario, epoch_num, prefix)


def _init_csv(file_path: str, header: list):
    """Clean and init summary csv file.

    Args:
        file_path (str): summary file path.
        header (list): expected header of summary file.
    """
    if os.path.exists(file_path):
        os.remove(file_path)
    with open(file_path, "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()


def _summary_append(
        scenario: GlobalScenarios, input_dir: str,
        file_name: str, header_list: list,
        sum_dataframe: pd.DataFrame, epoch_index: int, output_path: str):
    """Calculate summary info and generate corresponding csv file.

        To accelerate, change each column into numpy.array.

    Args:
        scenario (GlobalScenarios): Current scenario.
            This parameter is useless right now. Cause only scenario-cim needs operations within this function.
        input_dir (str): Current folder name.
            Loop is outside this function. This function calculate a summary within an epoch each time.
        file_name (str): Name of file needed to be summarized.
            Some scenario has multiple files within one epoch. e.g.cim.
        header_list (list): List of columns needed to be summarized.
        sum_dataframe (dataframe): Temporary dataframe to restore results.
        epoch_index (int): The epoch index of data being processing.
        output_path (str): Path of output CSV file.

    """
    input_path = os.path.join(input_dir, file_name)
    data = pd.read_csv(input_path)
    data_summary = []
    for header in header_list:
        data_summary.append(np.sum(np.array(data[header]), axis=0))
    sum_dataframe.loc[epoch_index] = data_summary
    sum_dataframe.to_csv(output_path, header=True, index=True)


def _generate_summary(scenario: GlobalScenarios, source_path: str, prefix: str, epoch_num: int):
    """Generate summary info of current scenario.

        Different scenario has different data features.
            e.g. cim has multiple epochs while citi_bike only has one.
        Each scenario should be treated respectively.

    Args:
        scenario (GlobalScenarios): Current scenario.
        source_path (str): The root path of the dumped snapshots data for the corresponding experiment.
        prefix (str): Prefix of data folders.
        epoch_num (int): Total number of epoches,
                        i.e. the total number of data folders since there is a folder per epoch.

    """
    ports_header = ["capacity", "empty", "full", "on_shipper", "on_consignee", "shortage", "booking", "fulfillment"]
    # vessels_header = ["capacity", "empty", "full", "remaining_space", "early_discharge"]
    stations_header = ["bikes", "shortage", "trip_requirement", "fulfillment", "capacity"]

    if scenario == GlobalScenarios.CIM:
        _init_csv(os.path.join(source_path, GlobalFilePaths.ports_sum), ports_header)
        # _init_csv(vessels_file_path, vessels_header)
        ports_sum_dataframe = pd.read_csv(
            os.path.join(source_path, GlobalFilePaths.ports_sum),
            names=ports_header
        )
        # vessels_sum_dataframe = pd.read_csv(vessels_file_path, names=vessels_header)
        for epoch_index in tqdm.tqdm(range(0, epoch_num)):
            epoch_folder = os.path.join(source_path, f"{prefix}{epoch_index}")
            input_dir = os.path.join(source_path, epoch_folder)
            _summary_append(
                scenario, input_dir, "ports.csv", ports_header,
                ports_sum_dataframe, epoch_index, os.path.join(source_path, GlobalFilePaths.ports_sum))
            # _summary_append(input_dir, "vessels.csv", vessels_header, vessels_sum_dataframe, i,vessels_file_path)
    elif scenario == GlobalScenarios.CITI_BIKE:
        _init_csv(os.path.join(source_path, GlobalFilePaths.stations_sum), stations_header)
        data = pd.read_csv(os.path.join(source_path, f"{prefix}0", "stations.csv"))
        data = data[["bikes", "trip_requirement", "fulfillment", "capacity"]].groupby(data["name"]).sum()
        data["fulfillment_ratio"] = list(
            map(
                lambda x, y: float("{:.4f}".format(x / (y + 1 / 1000))),
                data["fulfillment"],
                data["trip_requirement"]
                )
            )
        data.to_csv(os.path.join(source_path, GlobalFilePaths.stations_sum))


def _get_holder_name_conversion(scenario: GlobalScenarios, source_path: str, conversion_path: str):
    """ Generate a CSV File which indicates the relationship between resource holder's index and name.

    Args:
        scenario (GlobalScenarios): Current scenario. Different scenario has different type of mapping file.
        source_path (str): The root path of the dumped snapshots data for the corresponding experiment.
        conversion_path (str): Path of origin mapping file.
    """
    conversion_path = os.path.join(source_path, conversion_path)
    if os.path.exists(os.path.join(source_path, GlobalFilePaths.name_convert)):
        os.remove(os.path.join(source_path, GlobalFilePaths.name_convert))
    if scenario == GlobalScenarios.CITI_BIKE:
        with open(conversion_path, "r", encoding="utf8")as mapping_file:
            mapping_json_data = json.load(mapping_file)
            name_list = []
            for item in mapping_json_data["data"]["stations"]:
                name_list.append(item["name"])
            df = pd.DataFrame({"name": name_list})
            df.to_csv(os.path.join(source_path, GlobalFilePaths.name_convert), index=False)
    elif scenario == GlobalScenarios.CIM:
        mapping_file = open(conversion_path, "r")
        mapping_file_content = mapping_file.read()
        cim_information = yaml.load(mapping_file_content, Loader=yaml.FullLoader)
        conversion = cim_information["ports"].keys()
        df = pd.DataFrame(list(conversion))
        df.to_csv(os.path.join(source_path, GlobalFilePaths.name_convert), index=False)
