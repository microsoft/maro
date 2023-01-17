# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os

import requests

from .request_params import request_column, request_settings
from .utils import get_data_in_format, get_input_range


def get_new_port_number(experiment_name: str) -> json:
    """Get the latest episode number of real-time episode.

    Args:
        experiment_name (str): Name of the experiment.

    Returns:
            json: Number of episodes.

    """
    params = {
        "query": f"select count(episode) from {experiment_name}.port_details",
        "count": "true",
    }
    episode_number_data = requests.get(
        url=request_settings.request_url.value,
        headers=request_settings.request_header.value,
        params=params,
    ).json()
    return episode_number_data


def get_port_data(experiment_name: str, episode: str, tick: str) -> json:
    """Get the port data within one tick.

    Args:
        experiment_name (str): Name of the experiment expected to be displayed.
        episode (str) : Number of the episode of expected data.
        tick (str): Number of tick of expected data.

    Returns:
            json: Formatted port value of current tick.

    """
    params = {
        "query": f"select {request_column.port_header.value} from {experiment_name}.port_details"
        f" where episode='{episode}' and tick='{tick}'",
        "count": "true",
    }
    db_port_data = requests.get(
        url=request_settings.request_url.value,
        headers=request_settings.request_header.value,
        params=params,
    ).json()
    return process_port_data(db_port_data)


def get_acc_port_data(experiment_name: str, episode: str, start_tick: str, end_tick: str) -> json:
    """Get the port data within a range.

    Args:
        experiment_name (str): Name of the experiment expected to be displayed.
        episode (str) : Number of the episode of expected data.
        start_tick (str): Number of tick to the start point of port data.
        end_tick(str): Number of tick to the end point of port data.

    Returns:
            json: Jsonified formatted port value through a selected range.

    """
    input_range = get_input_range(start_tick, end_tick)
    query = (
        f"select {request_column.port_header.value}  from {experiment_name}.port_details" f" where episode='{episode}'"
    )
    if input_range != "()":
        query += f" and tick in {input_range}"
    params = {
        "query": query,
        "count": "true",
    }
    db_port_data = requests.get(
        url=request_settings.request_url.value,
        headers=request_settings.request_header.value,
        params=params,
    ).json()
    return process_port_data(db_port_data)


def process_port_data(db_port_data: json) -> json:
    """Generate compulsory columns and process with topoly information.

    Args:
        db_port_data(json): Original port data.

    Returns:
            json: Jsonfied port value of current tick.

    """
    exec_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
    config_file_path = f"{exec_path}/nginx/static/"
    with open(f"{config_file_path}port_list.json", "r", encoding="utf8") as port_list_file:
        port_list = json.load(port_list_file)
        port_list = port_list[0]["port_list"]
    with open(f"{config_file_path}port.json", "r", encoding="utf8") as port_file:
        port_json_data = json.load(port_file)

    original_port_data = get_data_in_format(db_port_data)
    original_port_data["port_name"] = list(
        map(
            lambda x: port_json_data[int(x)]["tooltip"],
            original_port_data["index"],
        ),
    )
    original_port_data["position"] = list(
        map(
            lambda x: port_json_data[int(x)]["position"],
            original_port_data["index"],
        ),
    )
    original_port_data["status"] = list(
        map(
            lambda x, y: "surplus" if (x - y * 5 > 50) else ("demand" if (x - y * 5 < -50) else "balance"),
            original_port_data["empty"],
            original_port_data["booking"],
        ),
    )
    port_data = original_port_data.to_json(orient="records")
    return port_data
