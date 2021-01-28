# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import csv
import json
import os
from typing import List

import numpy as np
import pandas as pd
import tqdm
import yaml

from maro.utils.exception.cli_exception import CliError
from maro.utils.logger import CliLogger

from .launch_env_dashboard import launch_dashboard
from .params import GlobalFileNames, GlobalScenarios

logger = CliLogger(name=__name__)


def start_vis(source_path: str, force: str, **kwargs: dict):
    """Entrance of data pre-processing.

    Generate index_name_conversion CSV file and summary file.

    Expected File Structure:
    -input_file_folder_path
        --epoch_0 : Data of current epoch.
            --holder_info.csv: Attributes of current epoch.
        ………………
        --epoch_{epoch_num-1}
        --manifest.yml: Record basic info like scenario name, name of index_name_mapping file.
        --index_name_mapping file: Record the relationship between an index and its name.
        Type of this file varied between scenario.

        Summary file would be generated after data processing.

    Args:
        source_path (str): Data folder path.
        force (str): Indicates whether regenerate data. Expected input is True/False.
        **kwargs (dict): The irrelevant variable length key-value pair.
    """

    if not os.path.exists(os.path.join(source_path, "manifest.yml")):
        raise CliError("Manifest file missed.")
    settings = yaml.load(
        open(os.path.join(source_path, "manifest.yml"), "r").read(),
        Loader=yaml.FullLoader
    )
    scenario = GlobalScenarios[str(settings["scenario"]).upper()]
    conversion_path = str(settings["mappings"])
    epoch_num = int(settings["dump_details"]["epoch_num"])
    prefix = settings["dump_details"]["prefix"]
    force = str2bool(str(force))
    if not os.path.exists(source_path):
        raise CliError("Input path is not correct.")
    elif not os.path.exists(os.path.join(source_path, f"{prefix}0")):
        raise CliError("No data under input folder path.")
    if force:
        logger.info("Generating Dashboard Data.")
        _get_index_index_name_conversion(scenario, source_path, conversion_path)
        logger.info_green("[1/2]:Generate Name Conversion File Done.")
        logger.info_green("[2/2]:Generating Summary.")
        _generate_summary(scenario, source_path, prefix, epoch_num)
        logger.info_green("[2/2]:Generate Summary Done.")

    else:
        logger.info_green("Skip Data Generation")
        if not os.path.exists(os.path.join(source_path, GlobalFileNames.name_convert)):
            raise CliError("Have to regenerate data. Name Conversion File is missed.")

        if scenario == GlobalScenarios.CIM:
            if not os.path.exists(os.path.join(source_path, GlobalFileNames.ports_sum)):
                raise CliError("Have to regenerate data. Summary File is missed.")
        elif scenario == GlobalScenarios.CITI_BIKE:
            if not os.path.exists(os.path.join(source_path, GlobalFileNames.stations_sum)):
                raise CliError("Have to regenerate data. Summary File is missed.")

    launch_dashboard(source_path, scenario, epoch_num, prefix)


def str2bool(force_type) -> bool:
    """Convert the parameter "force" from string to bool.

    Argsparse could not identify bool type automatically.
    Manually conversion is compulsory.

    Args:
        force_type (str): The parameter input by user.

    Returns:
        bool: Converted parameter.

    """
    if force_type.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif force_type.lower() in ('no', 'false', 'f', 'n', '0'):
        return False


def _init_csv(file_path: str, header: List[str]):
    """Clean and initiate summary csv file.

    This summary file record cross-epoch data.

    Args:
        file_path (str): Path of the summary file.
        header (List[str]): Expected header of summary file.
    """
    if os.path.exists(file_path):
        os.remove(file_path)
    with open(file_path, "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()


def _summary_append(
    scenario: GlobalScenarios, input_path: str, header_list: List[str],
    sum_dataframe: pd.DataFrame, epoch_index: int, output_path: str
):
    """Calculate summary info and generate corresponding csv file.

    To accelerate, change each column into numpy.array.

    Args:
        scenario (GlobalScenarios): Current scenario.
        input_path (str): Path of file needed to be summarized.
        header_list (List[str]): List of columns needed to be summarized.
        sum_dataframe (pd.Dataframe): Temporary dataframe to restore results.
        epoch_index (int): The epoch index of data being processing.
        output_path (str): Path of output CSV file.
    """
    data = pd.read_csv(input_path)
    data_summary = []
    for header in header_list:
        data_summary.append(np.sum(np.array(data[header]), axis=0))
    sum_dataframe.loc[epoch_index] = data_summary
    sum_dataframe.to_csv(output_path, header=True, index=True)


def _generate_summary(scenario: GlobalScenarios, source_path: str, prefix: str, epoch_num: int):
    """Generate summary info of current scenario.

    Different scenario has different data features.
    Each scenario should be treated respectively.

    Args:
        scenario (GlobalScenarios): Current scenario.
        source_path (str): The root path of the dumped snapshots data for the corresponding experiment.
        prefix (str): Prefix of data folders.
        epoch_num (int): Total number of epoches,
            i.e. the total number of data folders since there is a folder per epoch.
    """
    ports_header = ["capacity", "empty", "full", "on_shipper", "on_consignee", "shortage", "booking", "fulfillment"]
    stations_header = ["bikes", "shortage", "trip_requirement", "fulfillment", "capacity"]

    if scenario == GlobalScenarios.CIM:
        _init_csv(os.path.join(source_path, GlobalFileNames.ports_sum), ports_header)
        ports_sum_dataframe = pd.read_csv(
            os.path.join(source_path, GlobalFileNames.ports_sum),
            names=ports_header
        )
        for epoch_index in tqdm.tqdm(range(0, epoch_num)):
            input_path = os.path.join(source_path, f"{prefix}{epoch_index}", "ports.csv")
            _summary_append(
                scenario, input_path, ports_header,
                ports_sum_dataframe, epoch_index, os.path.join(source_path, GlobalFileNames.ports_sum)
            )
    elif scenario == GlobalScenarios.CITI_BIKE:
        _init_csv(os.path.join(source_path, GlobalFileNames.stations_sum), stations_header)
        stations_sum_dataframe = pd.read_csv(
            os.path.join(source_path, GlobalFileNames.stations_sum),
            names=stations_header
        )
        for epoch_index in tqdm.tqdm(range(0, epoch_num)):
            _init_csv(
                os.path.join(
                    source_path, f"{prefix}{epoch_index}", GlobalFileNames.stations_sum
                ),
                stations_header
            )
            input_path = os.path.join(source_path, f"{prefix}{epoch_index}", "stations.csv")
            data = pd.read_csv(input_path)
            data = data[["bikes", "trip_requirement", "fulfillment", "capacity"]].groupby(data["name"]).sum()
            data["fulfillment_ratio"] = list(
                map(
                    lambda x, y: round(x / (y + 1 / 1000), 4),
                    data["fulfillment"],
                    data["trip_requirement"]
                )
            )
            data.to_csv(os.path.join(source_path, f"{prefix}{epoch_index}", GlobalFileNames.stations_sum))
            _summary_append(
                scenario, input_path, stations_header, stations_sum_dataframe,
                epoch_index, os.path.join(source_path, GlobalFileNames.stations_sum)
            )


def _get_index_index_name_conversion(scenario: GlobalScenarios, source_path: str, conversion_path: str):
    """ Generate a CSV File which indicates the relationship between resource holder's index and name.

    Args:
        scenario (GlobalScenarios): Current scenario. Different scenario has different type of mapping file.
        source_path (str): The root path of the dumped snapshots data for the corresponding experiment.
        conversion_path (str): Path of original mapping file.
    """
    conversion_path = os.path.join(source_path, conversion_path)
    if os.path.exists(os.path.join(source_path, GlobalFileNames.name_convert)):
        os.remove(os.path.join(source_path, GlobalFileNames.name_convert))
    if scenario == GlobalScenarios.CITI_BIKE:
        with open(conversion_path, "r", encoding="utf8")as mapping_file:
            mapping_json_data = json.load(mapping_file)
            name_list = []
            for item in mapping_json_data["data"]["stations"]:
                name_list.append(item["name"])
            df = pd.DataFrame({"name": name_list})
            df.to_csv(os.path.join(source_path, GlobalFileNames.name_convert), index=False)
    elif scenario == GlobalScenarios.CIM:
        cim_information = yaml.load(
            open(conversion_path, "r").read(),
            Loader=yaml.FullLoader
        )
        conversion = cim_information["ports"].keys()
        df = pd.DataFrame(list(conversion))
        df.to_csv(os.path.join(source_path, GlobalFileNames.name_convert), index=False)
