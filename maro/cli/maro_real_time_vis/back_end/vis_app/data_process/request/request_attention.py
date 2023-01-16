# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pandas as pd
import requests

from .request_params import request_settings
from .utils import get_data_in_format


def get_attention_data(experiment_name: str, episode: str, tick: str) -> pd.DataFrame:
    """Get the attention data within one tick.

    Args:
        experiment_name (str): Name of the experiment expected to be displayed.
        episode (str) : Number of the episode of expected data.
        tick (str): Number of tick of expected data.

    Returns:
            Dataframe: Formatted attention value of current tick.

    """
    get_attention_value_params = {
        "query": f"select * from {experiment_name}.attentions where episode='{episode}' and tick='{tick}'",
        "count": "true",
    }
    attention_value = requests.get(
        url=request_settings.request_url.value,
        headers=request_settings.request_header.value,
        params=get_attention_value_params,
    ).json()
    attention_value = get_data_in_format(attention_value)
    return attention_value
