# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import time

import requests

from .request_params import request_settings
from .utils import get_data_in_format


def get_experiments():
    """Get a list of existing experiments.

    Returns:
        json: List of existing experiments.

    """
    get_exps_params = {
        "query": "select name from maro.experiments",
    }
    exps = requests.get(
        url=request_settings.request_url.value,
        headers=request_settings.request_header.value,
        params=get_exps_params,
    ).json()
    exp_name = json.dumps(exps["dataset"])
    return exp_name


def get_experiment_info():
    """Get basic information of experiment."""
    get_exp_name_params = {
        "query": "select name from pending_experiments order by time desc limit 1",
    }
    exp_name = requests.get(
        url=request_settings.request_url.value,
        headers=request_settings.request_header.value,
        params=get_exp_name_params,
    ).json()
    exp_name = exp_name["dataset"][0][0]
    params = {
        "query": f"select * from maro.experiments where name='{exp_name}'",
        "count": "true",
    }
    experiment_info = requests.get(
        url=request_settings.request_url.value,
        headers=request_settings.request_header.value,
        params=params,
    ).json()
    data_in_format = get_data_in_format(experiment_info)
    experiment_name = data_in_format["name"][0]
    episode_params = {
        "query": f"select episode, tick from {experiment_name}.port_details order by timestamp asc limit 1",
        "count": "true",
    }
    min_episode = requests.get(
        url=request_settings.request_url.value,
        headers=request_settings.request_header.value,
        params=episode_params,
    ).json()
    start_episode_num = int(min_episode["dataset"][0][0])
    start_snapshot_num = int(min_episode["dataset"][0][1])
    data_in_format["start_episode"] = start_episode_num
    data_in_format["start_snapshot"] = start_snapshot_num
    total_params = {
        "query": f"select count_distinct(episode), count_distinct(tick) from {experiment_name}.port_details",
        "count": "true",
    }
    total_episode = requests.get(
        url=request_settings.request_url.value,
        headers=request_settings.request_header.value,
        params=total_params,
    ).json()
    data_in_format["total_episodes"] = int(total_episode["dataset"][0][0])
    data_in_format["durations"] = int(total_episode["dataset"][0][1])
    port_number_params = {
        "query": f"select count(*) from {experiment_name}.port_details"
        f" where episode='{start_episode_num}' and tick='{start_snapshot_num}'",
        "count": "true",
    }
    port_number = requests.get(
        url=request_settings.request_url.value,
        headers=request_settings.request_header.value,
        params=port_number_params,
    ).json()
    end_epoch_num = start_episode_num + int(data_in_format["total_episodes"]) - 1
    end_tick_num = start_snapshot_num + int(total_episode["dataset"][0][1]) - 1
    display_type_params = {
        "query": f"select * from {experiment_name}.port_details"
        f" where episode='{end_epoch_num}' and tick='{end_tick_num}'",
        "count": "true",
    }
    display_type_response = requests.get(
        url=request_settings.request_url.value,
        headers=request_settings.request_header.value,
        params=display_type_params,
    ).json()
    if display_type_response["dataset"] != []:
        data_in_format["display_type"] = "local"
    else:
        data_in_format["display_type"] = "real_time"
    data_in_format["port_number"] = int(port_number["dataset"][0][0])
    exp_data = data_in_format.to_json(orient="records")
    return exp_data


def add_pending_experiment(experiment_name):
    current_time = int(time.time())
    next_exp_params = {
        "query": f"INSERT INTO pending_experiments(name, time) VALUES('{experiment_name}', {current_time})",
    }
    requests.get(
        url=request_settings.request_url.value,
        headers=request_settings.request_header.value,
        params=next_exp_params,
    ).json()
