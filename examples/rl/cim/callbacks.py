# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


def post_collect(info_list: list, ep: int, segment: int) -> None:
    # print the env metric from each rollout worker
    for info in info_list:
        print(f"env summary (episode {ep}, segment {segment}): {info['env_metric']}")

    # print the average env metric
    if len(info_list) > 1:
        metric_keys, num_envs = info_list[0]["env_metric"].keys(), len(info_list)
        avg_metric = {key: sum(info["env_metric"][key] for info in info_list) / num_envs for key in metric_keys}
        print(f"average env summary (episode {ep}, segment {segment}): {avg_metric}")


def post_evaluate(info_list: list, ep: int) -> None:
    # print the env metric from each rollout worker
    for info in info_list:
        print(f"env summary (episode {ep}): {info['env_metric']}")

    # print the average env metric
    if len(info_list) > 1:
        metric_keys, num_envs = info_list[0]["env_metric"].keys(), len(info_list)
        avg_metric = {key: sum(info["env_metric"][key] for info in info_list) / num_envs for key in metric_keys}
        print(f"average env summary (episode {ep}): {avg_metric}")
