# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
from typing import Union

import pandas as pd


def get_data_in_format(original_data: Union[int, slice]) -> pd.DataFrame:
    """Convert the json data into dataframe.

    Args:
        original_data (Union[int, slice]): Json data requested from database directly.

    Returns:
        Dataframe: Formatted dataframe.

    """
    dataset = original_data["dataset"]
    column = original_data["columns"]
    dataheader = []
    for col_index in range(0, len(column)):
        dataheader.append(column[col_index]["name"])
    data_in_format = pd.DataFrame(dataset, columns=dataheader)
    return data_in_format


def get_input_range(start_tick: str, end_tick: Union[int, slice]) -> Union[int, slice]:
    """Get the tick input range in string format.

    Args:
        start_tick(str): Start point of range.
        end_tick(str): End point of range.

    Returns:
        Union[int, slice]: Range of tick in string format.

    """
    i = int(start_tick)
    input_range = "("
    while i < end_tick:
        if i == int(end_tick) - 1:
            input_range = input_range + "'" + str(i) + "'" + ")"
        else:
            input_range = input_range + "'" + str(i) + "'" + ","
        i = i + 1
    return input_range
