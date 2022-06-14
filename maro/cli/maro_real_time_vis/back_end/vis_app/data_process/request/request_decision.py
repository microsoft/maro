# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json

import pandas as pd
import requests

from .request_params import request_column, request_settings
from .utils import get_data_in_format, get_input_range


def get_decision_data(experiment_name: str, episode: str, tick: str) -> pd.DataFrame:
    """Get the decision data within one tick.

    Args:
        experiment_name (str): Name of the experiment expected to be displayed.
        episode (str) : Number of the episode of expected data.
        tick (str): Number of tick of expected data.

    Returns:
            Dataframe: Formatted decision value of current tick.

    """
    params = {
        "query": f"select {request_column.decision_header.value} from {experiment_name}.full_on_vessels"
        f" where episode='{episode}' and tick='{tick}'",
        "count": "true",
    }
    decision_value = requests.get(
        url=request_settings.request_url.value,
        headers=request_settings.request_header.value,
        params=params,
    ).json()
    decision_data = get_data_in_format(decision_value).to_json(orient="records")
    return decision_data


def get_acc_decision_data(experiment_name: str, episode: str, start_tick: str, end_tick: str) -> json:
    """Get the decision data within a range.

    Args:
        experiment_name (str): Name of the experiment expected to be displayed.
        episode (str) : Number of the episode of expected data.
        start_tick (str): Number of tick to the start point of decision data.
        end_tick(str): Number of tick to the end point of decision data.

    Returns:
            json: Jsonified formatted decision value through a selected range.

    """
    input_range = get_input_range(start_tick, end_tick)
    query = (
        f"select {request_column.decision_header.value} from {experiment_name}.full_on_vessels"
        f" where episode='{episode}'"
    )
    if input_range != "()":
        query += f" and tick in {input_range}"
    params = {
        "query": query,
        "count": "true",
    }
    original_decision_data = requests.get(
        url=request_settings.request_url.value,
        headers=request_settings.request_header.value,
        params=params,
    ).json()
    decision_data = get_data_in_format(original_decision_data)
    decision_output = []
    i = start_tick
    while i < end_tick:
        cur_decision = decision_data[decision_data["tick"] == str(i)].copy()
        if cur_decision.empty:
            decision_output.append([])
        else:
            decision_in_format = cur_decision.to_json(orient="records")
            decision_output.append(json.loads(decision_in_format))
        i = i + 1
    return decision_output
