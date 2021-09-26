# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
from os import makedirs
from os.path import dirname, join, realpath

from matplotlib import pyplot as plt

timestamp = str(time.time())
log_dir = join(dirname(realpath(__file__)), "log", timestamp)
makedirs(log_dir, exist_ok=True)
plt_path = join(dirname(realpath(__file__)), "plots", timestamp)
makedirs(plt_path, exist_ok=True)


def post_collect(trackers, ep, segment):
    # print the env metric from each rollout worker
    for tracker in trackers:
        print(f"env summary (episode {ep}, segment {segment}): {tracker['env_metric']}")

    # print the average env metric
    if len(trackers) > 1:
        metric_keys, num_trackers = trackers[0]["env_metric"].keys(), len(trackers)
        avg_metric = {key: sum(tr["env_metric"][key] for tr in trackers) / num_trackers for key in metric_keys}
        print(f"average env metric (episode {ep}, segment {segment}): {avg_metric}")


def post_evaluate(trackers, ep):
    # print the env metric from each rollout worker
    for tracker in trackers:
        print(f"env summary (evaluation episode {ep}): {tracker['env_metric']}")

    # print the average env metric
    if len(trackers) > 1:
        metric_keys, num_trackers = trackers[0]["env_metric"].keys(), len(trackers)
        avg_metric = {key: sum(tr["env_metric"][key] for tr in trackers) / num_trackers for key in metric_keys}
        print(f"average env metric (evaluation episode {ep}): {avg_metric}")

    for tracker in trackers:
        core_requirement = tracker["actions_by_core_requirement"]
        action_sequence = tracker["action_sequence"]
        # plot action sequence
        fig = plt.figure(figsize=(40, 32))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(action_sequence)
        fig.savefig(f"{plt_path}/action_sequence_{ep}")
        plt.cla()
        plt.close("all")

        # plot with legal action mask
        fig = plt.figure(figsize=(40, 32))
        for idx, key in enumerate(core_requirement.keys()):
            ax = fig.add_subplot(len(core_requirement.keys()), 1, idx + 1)
            for i in range(len(core_requirement[key])):
                if i == 0:
                    ax.plot(core_requirement[key][i][0] * core_requirement[key][i][1], label=str(key))
                    ax.legend()
                else:
                    ax.plot(core_requirement[key][i][0] * core_requirement[key][i][1])

        fig.savefig(f"{plt_path}/values_with_legal_action_{ep}")

        plt.cla()
        plt.close("all")

        # plot without legal actin mask
        fig = plt.figure(figsize=(40, 32))

        for idx, key in enumerate(core_requirement.keys()):
            ax = fig.add_subplot(len(core_requirement.keys()), 1, idx + 1)
            for i in range(len(core_requirement[key])):
                if i == 0:
                    ax.plot(core_requirement[key][i][0], label=str(key))
                    ax.legend()
                else:
                    ax.plot(core_requirement[key][i][0])

        fig.savefig(f"{plt_path}/values_without_legal_action_{ep}")

        plt.cla()
        plt.close("all")
