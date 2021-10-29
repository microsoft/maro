# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

attributes = ["kw", "dat", "at", "sps", "das", "total_kw"]

def get_baseline():
    data_path = "/home/Jinyu/maro/maro/simulator/scenarios/hvac/topologies/building121/datasets/train_data_AHU_MAT.csv"
    df = pd.read_csv(data_path, sep=',', delimiter=None, header='infer')
    df = df.dropna()
    df = df.reset_index()

    return {
        "kw": df["KW"].to_numpy(),
        "dat": df["DAT"].to_numpy(),
        "at": df["air_ton"].to_numpy(),
        # "mat": df["DAS"].to_numpy() + df["delta_MAT_DAS"].to_numpy(),
        "sps": df["SPS"].to_numpy(),
        "das": df["DAS"].to_numpy(),
        "total_kw": np.cumsum(df["KW"].to_numpy())
    }

baseline = get_baseline()

def post_evaluate(trackers: dict, episode: int, path: str, prefix: str="Eval"):
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))

    for idx, att in enumerate(attributes):
        axs[idx//3, idx%3].plot(trackers[att][1:], c='r')
        axs[idx//3, idx%3].plot(baseline[att][:len(trackers[att][1:])], c='b')
        axs[idx//3, idx%3].set_title(att)

    fig.savefig(os.path.join(path, f"{prefix}_{episode}.png"))
    plt.close(fig)

def post_collect(trackers: dict, episode: int, path: str):
    post_evaluate(trackers, episode, path, prefix="Train")
