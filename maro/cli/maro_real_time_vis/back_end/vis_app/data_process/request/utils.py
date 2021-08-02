# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pandas as pd


def get_data_in_format(original_data) -> pd.DataFrame:
    """Convert the json data into dataframe.

    Args:
        original_data: Json data requested from database directly.

    Returns:
        pd.Dataframe: Formatted dataframe.

    """
    dataset = original_data["dataset"]
    column = original_data["columns"]
    dataheader = []
    for col_index in range(0, len(column)):
        dataheader.append(column[col_index]["name"])
    data_in_format = pd.DataFrame(dataset, columns=dataheader)
    return data_in_format


def get_input_range(start_tick: str, end_tick: str) -> str:
    """Get the tick input range in string format.

    Args:
        start_tick(str): Start point of range.
        end_tick(str): End point of range.

    Returns:
        str: Range of tick in string format.

    """
    return "(" + ", ".join([f"'{i}'" for i in range(int(start_tick), int(end_tick))]) + ")"
