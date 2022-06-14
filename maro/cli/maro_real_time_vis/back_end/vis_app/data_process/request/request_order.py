# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json

import pandas as pd
import requests

from .request_params import request_column, request_settings
from .utils import get_data_in_format, get_input_range


def get_order_data(experiment_name: str, episode: str, tick: str) -> pd.DataFrame:
    """Get the order data within one tick.

    Args:
        experiment_name (str): Name of the experiment expected to be displayed.
        episode (str) : Number of the episode of expected data.
        tick (str): Number of tick of expected data.

    Returns:
            Dataframe: Formatted order value of current tick.

    """
    params = {
        "query": f"select {request_column.order_header.value} from {experiment_name}.full_on_ports"
        f" where episode='{episode}' and tick='{tick}'",
        "count": "true",
    }
    original_order_data = requests.get(
        url=request_settings.request_url.value,
        headers=request_settings.request_header.value,
        params=params,
    ).json()
    order_data = get_data_in_format(original_order_data).to_json(orient="records")
    return order_data


def get_acc_order_data(experiment_name: str, episode: str, start_tick: str, end_tick: str) -> json:
    """Get the order data within a range.

    Args:
        experiment_name (str): Name of the experiment expected to be displayed.
        episode (str) : Number of the episode of expected data.
        start_tick (str): Number of tick to the start point of order data.
        end_tick(str): Number of tick to the end point of order data.

    Returns:
            json: Jsonified formatted order value through a selected range.

    """
    input_range = get_input_range(start_tick, end_tick)
    query = (
        f"select {request_column.order_header.value} from {experiment_name}.full_on_ports" f" where episode='{episode}'"
    )
    if input_range != "()":
        query += f" and tick in {input_range}"
    params = {
        "query": query,
        "count": "true",
    }
    original_order_data = requests.get(
        url=request_settings.request_url.value,
        headers=request_settings.request_header.value,
        params=params,
    ).json()
    order_data = get_data_in_format(original_order_data)
    order_output = []
    i = start_tick
    while i < end_tick:
        cur_order = order_data[order_data["tick"] == str(i)].copy()
        if cur_order.empty:
            order_output.append([])
        else:
            order_in_format = cur_order.to_json(orient="records")
            order_output.append(json.loads(order_in_format))
        i = i + 1
    return order_output
