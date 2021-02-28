import pandas as pd
import requests

from .request_params import request_settings


def get_attention_data(experiment_name, episode, tick):
    get_attention_value_params = {
        "query": f"select * from {experiment_name}.attentions where episode='{episode}' and tick='{tick}'",
        "count": "true"
    }
    attention_value = requests.get(request_settings.request_url.value, headers=request_settings.request_header.value, params=get_attention_value_params).json()
    if attention_value['dataset'] != []:
        dataset = attention_value["dataset"]
        column = attention_value["columns"]
        dataheader = []
        for col_index in range(0, len(column)):
            dataheader.append(column[col_index]["name"])
        original_attention_data = pd.DataFrame(dataset, columns=dataheader)
        print(original_attention_data)
    return attention_value