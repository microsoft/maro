import requests
import json

from .utils import get_data_in_format
from .request_params import request_settings


def get_experiment_info():
    # host.docker.internal

    params = {
        "query": "select * from maro.experiments order by timestamp desc limit 1",
        "count": "true"
    }
    requests.DEFAULT_RETRIES = 5
    s = requests.session()
    s.keep_alive = False
    experiment_info = requests.get(request_settings.request_url.value, headers=request_settings.request_header.value, params=params).json()
    data_in_format = get_data_in_format(experiment_info)
    experiment_name = data_in_format["name"][0]
    episode_params = {
        "query": f"select episode, tick from {experiment_name}.port_details order by timestamp asc limit 1",
        "count": "true"
    }
    min_episode = requests.get(request_settings.request_url.value, headers=request_settings.request_header.value, params=episode_params).json()
    start_episode_num = int(min_episode["dataset"][0][0])
    start_snapshot_num = int(min_episode["dataset"][0][1])
    data_in_format["start_episode"] = start_episode_num
    data_in_format['start_snapshot'] = start_snapshot_num
    total_params = {
        "query": f"select count(episode), count(tick) from {experiment_name}.port_details",
        "count": "true"
    }
    total_episode = requests.get(request_settings.request_url.value, headers=request_settings.request_header.value, params=total_params).json()
    data_in_format["total_episodes"] = int(total_episode["dataset"][0][0])
    data_in_format['durations'] = int(total_episode["dataset"][0][1])
    port_number_params = {
        "query": f"select count(*) from {experiment_name}.port_details where episode='{start_episode_num}' and tick='{start_snapshot_num}'",
        "count": "true"
    }
    port_number = requests.get(
        url=request_settings.request_url.value,
        headers=request_settings.request_header.value,
        params=port_number_params
    ).json()
    data_in_format["port_number"] = int(port_number["dataset"][0][0])
    exp_data = data_in_format.to_json(orient='records')
    return exp_data


def get_min_episode_snapshot(experiment_name):
    episode_params = {
        "query": f"select episode from {experiment_name}.port_details order by timestamp asc limit 1",
        "count": "true"
    }
    snapshot_params = {
        "query": f"select tick from {experiment_name}.port_details order by timestamp asc limit 1",
        "count": "true"
    }
    requests.DEFAULT_RETRIES = 5
    s = requests.session()
    s.keep_alive = False
    min_episode = requests.get(request_settings.request_url.value, headers=request_settings.request_header.value, params=episode_params).json()
    min_snapshot = requests.get(request_settings.request_url.value, headers=request_settings.request_header.value, params=snapshot_params).json()
    print(min_episode)
    return json.dumps({"min_episode": int(min_episode["dataset"][0][0]), "min_snapshot": int(min_snapshot["dataset"][0][0])})


