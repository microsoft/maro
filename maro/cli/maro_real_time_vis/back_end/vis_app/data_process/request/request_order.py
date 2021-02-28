import requests
import json

from .utils import get_data_in_format, get_input_range
from .request_params import request_column, request_settings


def get_order_data(experiment_name, episode, tick):
    params = {
        "query": f"select {request_column.order_header.value} from {experiment_name}.full_on_ports where episode='{episode}' and tick='{tick}'",
        "count": "true"
    }                                                                                           
    original_order_data = requests.get(request_settings.request_url.value, headers=request_settings.request_header.value, params=params).json()
    order_data = get_data_in_format(original_order_data).to_json(orient='records')
    return order_data


def get_acc_order_data(experiment_name, episode, start_tick, end_tick):
    input_range = get_input_range(start_tick, end_tick)
    params = {
        "query": f"select {request_column.order_header.value} from {experiment_name}.full_on_ports where episode='{episode}' and tick in {input_range}",
        "count": "true"
    }
    original_order_data = requests.get(request_settings.request_url.value, headers=request_settings.request_header.value, params=params).json()
    order_data = get_data_in_format(original_order_data)
    order_output = []
    i = start_tick
    while i < end_tick:
        cur_order = order_data[order_data["tick"] == str(i)].copy()
        if cur_order.empty:
            order_output.append([])
        else:
            order_in_format = cur_order.to_json(orient='records')
            order_output.append(json.loads(order_in_format))
        i = i + 1
    return order_output

