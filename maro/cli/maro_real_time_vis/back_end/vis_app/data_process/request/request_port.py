import requests
import pandas as pd
import json

from .utils import get_data_in_format, get_input_range
from .request_params import request_column, request_settings


def get_port_data(experiment_name, episode, tick):
    params = {
        "query": f"select {request_column.port_header.value} from {experiment_name}.port_details where episode='{episode}' and tick='{tick}'",
        "count": "true"
    }                                                                                                                                              
    db_port_data = requests.get(request_settings.request_url.value, headers=request_settings.request_header.value, params=params).json()
    return process_port_data(db_port_data)


def get_acc_port_data(experiment_name, episode, start_tick, end_tick):
    input_range = get_input_range(start_tick, end_tick)
    params = {
        "query": f"select {request_column.port_header.value}  from {experiment_name}.port_details where episode='{episode}' and tick in {input_range}",
        "count": "true"
    }                                                                                                                                    
    db_port_data = requests.get(request_settings.request_url.value, headers=request_settings.request_header.value, params=params).json()
    return process_port_data(db_port_data)


def process_port_data(db_port_data):
    with open(r"../nginx/static/port_list.json", "r", encoding="utf8")as port_list_file:
        port_list = json.load(port_list_file)
        port_list = port_list[0]["port_list"]
    with open(r"../nginx/static/port.json", "r", encoding="utf8")as port_file:
        port_json_data = json.load(port_file)

    original_port_data = get_data_in_format(db_port_data)
    original_port_data["port_name"] = list(
            map(
                lambda x: port_json_data[int(x)]['tooltip'],
                original_port_data["index"]
            )
        )
    original_port_data["position"] = list(
        map(
            lambda x: port_json_data[int(x)]['position'],
            original_port_data["index"]
        )
    )
    original_port_data["status"] = list(
        map(
            lambda x, y: 'surplus' if (x - y*5 > 50) else ('demand' if (x - y*5 < -50) else 'balance'),
            original_port_data['empty'], original_port_data['booking']
        )
    )
    port_data = original_port_data.to_json(orient='records')
    return port_data

