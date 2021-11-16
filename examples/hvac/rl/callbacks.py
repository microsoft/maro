# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import config

attributes = ["kw", "dat", "at", "sps", "das", "total_kw"]

def get_baseline(baseline_path):
    df = pd.read_csv(baseline_path, sep=',', delimiter=None, header='infer')
    df = df.dropna()
    df = df.reset_index()

    return {
        "kw": df["KW"].to_numpy(),
        "dat": df["DAT"].to_numpy(),
        "at": df["air_ton"].to_numpy(),
        "mat": df["DAS"].to_numpy() + df["delta_MAT_DAS"].to_numpy(),
        "sps": df["SPS"].to_numpy(),
        "das": df["DAS"].to_numpy(),
        "total_kw": np.cumsum(df["KW"].to_numpy())
    }

baseline = get_baseline(config.baseline_path)

def post_evaluate(trackers: dict, episode: int, path: str, prefix: str="Eval"):

    def get_title(att: str):
        data = trackers[att]
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        if "total" in att:
            if "kw" in att:
                baseline_total = np.max(baseline[att][:len(data)])
                return f"{att}_[{np.min(data):.5f}, {np.max(data):.5f}]_{(baseline_total - np.max(data))/np.max(data):.2%}"
            return f"{att}_[{np.min(data):.5f}, {np.max(data):.5f}]"
        return f"{att}_[{np.min(data):.3f}, {np.max(data):.3f}]_({np.mean(data):.3f}, {np.std(data):.3f})"

    fig_plot, axs_plot = plt.subplots(2, 4, figsize=(20, 9))
    fig_hist, axs_hist = plt.subplots(2, 3, figsize=(20, 9))

    for idx, att in enumerate(attributes):
        axs_plot[idx//3, idx%3].plot(trackers[att], c='r')
        axs_plot[idx//3, idx%3].plot(baseline[att][:len(trackers[att])], c='b')
        axs_plot[idx//3, idx%3].set_title(get_title(att))

        if "total" in att:
            continue

        bins = np.linspace(
            min(min(trackers[att]), min(baseline[att])),
            max(max(trackers[att]), max(baseline[att])),
            15
        )
        axs_hist[idx//3, idx%3].hist(trackers[att], bins=bins, density=True, color='r', alpha=0.4)
        axs_hist[idx//3, idx%3].hist(baseline[att], bins=bins, density=True, color='b', alpha=0.4)
        axs_hist[idx//3, idx%3].set_title(att)

    axs_plot[0, 3].plot(trackers["reward"], c='r')
    axs_plot[0, 3].set_title(get_title("reward"))

    axs_plot[1, 3].plot(trackers["total_reward"], c='r')
    axs_plot[1, 3].set_title(get_title("total_reward"))

    axs_hist[1, 2].hist(trackers["reward"], bins=15, density=True, color='r', alpha=0.4)
    axs_hist[1, 2].set_title("reward")

    fig_plot.savefig(os.path.join(path, f"{prefix}_{episode}_plot.png"))
    plt.close(fig_plot)

    fig_hist.savefig(os.path.join(path, f"{prefix}_{episode}_hist.png"))
    plt.close(fig_hist)

    with open(os.path.join(path, f"data_{prefix}_{episode}.csv"), 'w') as fp:
        writer = csv.writer(fp)
        headers = ["kw", "dat", "at", "mat", "sps", "das", "total_kw", "reward", "total_reward"]
        writer.writerow(headers)

        rows = [
            [trackers[key][i] for key in headers]
            for i in range(len(trackers["kw"]))
        ]
        writer.writerows(rows)

    data = trackers['total_kw']
    res = res = f"{(np.max(baseline['total_kw'][:len(data)]) - np.max(data))/np.max(data):.2%}"
    return res

def post_collect(trackers: dict, episode: int, path: str):
    post_evaluate(trackers, episode, path, prefix="Train")

def visualize_returns(returns, log_dir, title):
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    x = list(range(len(returns)))
    ax.plot(x, returns, color="red")

    ax.set_xlabel('Episode')
    ax.set_xlim([0, x[-1]])
    ax.set_ylabel('Return')
    ax.set_ylim([np.min(returns), np.max(returns)])

    ax.set_title(title)
    plt.savefig(os.path.join(log_dir, f"{title}.png"))
    plt.clf()
