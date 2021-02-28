import requests
import json

from .utils import get_input_range, get_data_in_format
from .request_params import request_column, request_settings


def get_decision_data(experiment_name, episode, tick):
    params = {
        "query": f"select {request_column.decision_header.value} from {experiment_name}.full_on_vessels where episode='{episode}' and tick='{tick}'",
        "count": "true"
    }
    decision_value = requests.get(request_settings.request_url.value, headers=request_settings.request_header.value, params=params).json()
    decision_data = get_data_in_format(decision_value).to_json(orient='records')
    return decision_data


def get_acc_decision_data(experiment_name, episode, start_tick, end_tick):
    input_range = get_input_range(start_tick, end_tick)
    params = {
        "query": f"select {request_column.decision_header.value} from {experiment_name}.full_on_vessels where episode='{episode}' and tick in {input_range}",
        "count": "true"
    }
    original_decision_data = requests.get(request_settings.request_url.value, headers=request_settings.request_header.value, params=params).json()
    decision_data = get_data_in_format(original_decision_data)
    decision_output = []
    i = start_tick
    while i < end_tick:
        cur_decision = decision_data[decision_data["tick"] == str(i)].copy()
        if cur_decision.empty:
            decision_output.append([])
        else:
            decision_in_format = cur_decision.to_json(orient='records')
            decision_output.append(json.loads(decision_in_format))
        i = i + 1
    return decision_output
