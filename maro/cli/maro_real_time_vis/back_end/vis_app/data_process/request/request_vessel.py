# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os

import pandas as pd
import requests
from flask import jsonify

from .request_params import request_column, request_settings
from .utils import get_data_in_format, get_input_range


def get_vessel_data(experiment_name: str, episode: str, tick: str) -> json:
    """Get the vessel data within one tick.

    Args:
        experiment_name (str): Name of the experiment expected to be displayed.
        episode (str) : Number of the episode of expected data.
        tick (str): Number of tick of expected data.

    Returns:
            json: Jsonfied vessel value of current tick.

    """
    params = {
        "query": f"select {request_column.vessel_header.value} from {experiment_name}.vessel_details"
        f" where episode='{episode}' and tick='{tick}'",
        "count": "true",
    }
    db_vessel_data = requests.get(
        url=request_settings.request_url.value,
        headers=request_settings.request_header.value,
        params=params,
    ).json()
    return jsonify(process_vessel_data(db_vessel_data, tick))


def get_acc_vessel_data(experiment_name: str, episode: str, start_tick: str, end_tick: str) -> json:
    """Get the vessel data within a range.

    Args:
        experiment_name (str): Name of the experiment expected to be displayed.
        episode (str) : Number of the episode of expected data.
        start_tick (str): Number of tick to the start point of vessel data.
        end_tick(str): Number of tick to the end point of vessel data.

    Returns:
            json: Jsonified formatted vessel value through a selected range.

    """
    input_range = get_input_range(start_tick, end_tick)
    query = (
        f"select {request_column.vessel_header.value} from {experiment_name}.vessel_details"
        f" where episode='{episode}'"
    )
    if input_range != "()":
        query += f" and tick in {input_range}"
    params = {
        "query": query,
        "count": "true",
    }
    db_vessel_data = requests.get(
        url=request_settings.request_url.value,
        headers=request_settings.request_header.value,
        params=params,
    ).json()
    return jsonify(process_vessel_data(db_vessel_data, start_tick))


def process_vessel_data(db_vessel_data: json, start_tick: str) -> json:
    """Process the vessel data with route information.

    Args:
        db_vessel_data(json): Original vessel data.
            Both accumulated data and single data are possible.
        start_tick(str): Number of first tick of data.

    Returns:
            json: Jsonified formatted vessel value.

    """
    exec_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
    config_file_path = f"{exec_path}/nginx/static/config.json"
    with open(config_file_path, "r") as mapping_file:
        cim_information = json.load(mapping_file)
        vessel_list = list(cim_information["vessels"].keys())
        vessel_info = cim_information["vessels"]
        route_list = list(cim_information["routes"].keys())
    for item in route_list:
        route_distance = cim_information["routes"][item]
        route_distance_length = len(route_distance)
        prev = 0
        route_distance[0]["distance_to_next_port"] = 0
        for index in range(1, route_distance_length):
            route_distance[index]["distance_to_next_port"] = route_distance[index]["distance_to_next_port"] + prev
            prev = route_distance[index]["distance_to_next_port"]
    original_vessel_data = get_data_in_format(db_vessel_data)

    frame_index_num = len(original_vessel_data["tick"].unique())
    if frame_index_num == 1:
        return get_single_snapshot_vessel_data(
            original_vessel_data,
            vessel_list,
            vessel_info,
            route_list,
            cim_information,
        )
    else:
        acc_vessel_data = []
        for vessel_index in range(0, frame_index_num):
            cur_vessel_data = original_vessel_data[
                original_vessel_data["tick"] == str(vessel_index + start_tick)
            ].copy()
            acc_vessel_data.append(
                get_single_snapshot_vessel_data(
                    cur_vessel_data,
                    vessel_list,
                    vessel_info,
                    route_list,
                    cim_information,
                ),
            )
        return acc_vessel_data


def get_single_snapshot_vessel_data(
    original_vessel_data: pd.DataFrame,
    vessel_list: list,
    vessel_info: json,
    route_list: list,
    cim_information: json,
):
    """Generate compulsory data and change vessel data format.

    Args:
        original_vessel_data(DataFrame): Vessel data without generated columns.
        vessel_list(list): List of vessel of current topology.
        vessel_info(json): Vessel detailed information.
        route_list(list): List of route of current topology.
        cim_information(json): Topology information.

    Returns:
            json: Jsonified formatted vessel value.

    """
    original_vessel_data["name"] = list(
        map(
            lambda x: vessel_list[int(x)],
            original_vessel_data["index"],
        ),
    )
    original_vessel_data["speed"] = list(
        map(
            lambda x: vessel_info[x]["sailing"]["speed"],
            original_vessel_data["name"],
        ),
    )
    original_vessel_data["route name"] = list(
        map(
            lambda x: vessel_info[x]["route"]["route_name"],
            original_vessel_data["name"],
        ),
    )
    original_vessel_data["start port"] = list(
        map(
            lambda x: vessel_info[x]["route"]["initial_port_name"],
            original_vessel_data["name"],
        ),
    )
    original_vessel_data["start"] = 0
    vessel_data = original_vessel_data.to_json(orient="records")
    vessel_json_data = json.loads(vessel_data)
    output = []
    for item in route_list:
        vessel_in_output = []
        for vessel in vessel_json_data:
            if vessel["route name"] == item:
                start_port = vessel["start port"]
                route_distance_info = cim_information["routes"][item]
                for dis in route_distance_info:
                    if dis["port_name"] == start_port:
                        vessel["start"] = dis["distance_to_next_port"]
                vessel_in_output.append(vessel)
        output.append({"name": item, "vessel": vessel_in_output})

    return output
