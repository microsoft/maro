# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
from os import makedirs
from os.path import dirname, join, realpath

log_dir = join(dirname(realpath(__file__)), "log", str(time.time()))
makedirs(log_dir, exist_ok=True)


def post_collect(trackers, ep, segment):
    # print the env metric from each rollout worker
    for tracker in trackers:
        print(f"env summary (episode {ep}, segment {segment}): {tracker['env_metric']}")

    # print the average env metric
    if len(trackers) > 1:
        metric_keys, num_trackers = trackers[0]["env_metric"].keys(), len(trackers)
        avg_metric = {key: sum(tr["env_metric"][key] for tr in trackers) / num_trackers for key in metric_keys}
        print(f"average env summary (episode {ep}, segment {segment}): {avg_metric}")


def post_evaluate(trackers, ep):
    # print the env metric from each rollout worker
    for tracker in trackers:
        print(f"env summary (episode {ep}): {tracker['env_metric']}")

    # print the average env metric
    if len(trackers) > 1:
        metric_keys, num_trackers = trackers[0]["env_metric"].keys(), len(trackers)
        avg_metric = {key: sum(tr["env_metric"][key] for tr in trackers) / num_trackers for key in metric_keys}
        print(f"average env summary (episode {ep}): {avg_metric}")
