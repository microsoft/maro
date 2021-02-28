import requests
import pandas as pd
request_url = "http://127.0.0.1:9000/exec"
# host.docker.internal
request_header = {
        'content-type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'PUT,GET,POST,DELETE,OPTIONS',
        'Cache-Control': 'no-cache, no-transform'
}
params = {
    "query": "select * from maro.experiments order by timestamp desc limit 1",
    "count": "true"
}
requests.DEFAULT_RETRIES = 5
s = requests.session()
s.keep_alive = False
experiment_info = requests.get(request_url, headers=request_header, params=params).json()
dataset = experiment_info["dataset"]
column = experiment_info["columns"]
dataheader = []
for col_index in range(0, len(column)):
    dataheader.append(column[col_index]["name"])
data_in_format = pd.DataFrame(dataset, columns=dataheader)
experiment_name = data_in_format["name"][0]
episode_params = {
    "query": f"select episode, tick from {experiment_name}.port_details order by timestamp asc limit 1",
    "count": "true"
}
min_episode = requests.get(request_url, headers=request_header, params=episode_params).json()
print(min_episode)
data_in_format["start_episode"] = int(min_episode["dataset"][0][0])
data_in_format['start_snapshot'] = int(min_episode["dataset"][0][1])
print(data_in_format)