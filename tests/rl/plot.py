# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOG_DIR = "tests/rl/log"

color_map = {
    "ppo": "green",
    "sac": "goldenrod",
    "ddpg": "firebrick",
    "vpg": "cornflowerblue",
    "td3": "mediumpurple",
}


def smooth(data: np.ndarray, window_size: int) -> np.ndarray:
    if window_size > 1:
        """
        smooth data with moving window average.
        that is,
            smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
        where the "smooth" param is width of that window (2k+1)
        """
        y = np.ones(window_size)
        x = np.asarray(data)
        z = np.ones_like(x)
        smoothed_x = np.convolve(x, y, "same") / np.convolve(z, y, "same")
        return smoothed_x
    else:
        return data


def get_off_policy_data(log_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    file_path = os.path.join(log_dir, "metrics_full.csv")
    df = pd.read_csv(file_path)
    x, y = df["n_interactions"], df["val/avg_reward"]
    mask = ~np.isnan(y)
    x, y = x[mask], y[mask]
    return x, y


def get_on_policy_data(log_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    file_path = os.path.join(log_dir, "metrics_full.csv")
    df = pd.read_csv(file_path)
    x, y = df["n_interactions"], df["avg_reward"]
    return x, y


def plot_performance_curves(title: str, dir_names: List[str], smooth_window_size: int) -> None:
    for algorithm in color_map.keys():
        if algorithm in ["ddpg", "sac", "td3"]:
            func = get_off_policy_data
        elif algorithm in ["ppo", "vpg"]:
            func = get_on_policy_data

        log_dirs = [os.path.join(LOG_DIR, name) for name in dir_names if algorithm in name]
        series = [func(log_dir) for log_dir in log_dirs if os.path.exists(log_dir)]
        if len(series) == 0:
            continue

        x = series[0][0]
        assert all(len(_x) == len(x) for _x, _ in series), f"Input data should share the same length!"
        ys = np.array([smooth(y, smooth_window_size) for _, y in series])
        y_mean = np.mean(ys, axis=0)
        y_std = np.std(ys, axis=0)

        plt.plot(x, y_mean, label=algorithm, color=color_map[algorithm])
        plt.fill_between(x, y_mean - y_std, y_mean + y_std, color=color_map[algorithm], alpha=0.2)

    plt.legend()
    plt.grid()
    plt.title(title)
    plt.xlabel("Total Env Interactions")
    plt.ylabel(f"Average Trajectory Return \n(moving average with window size = {smooth_window_size})")
    plt.savefig(os.path.join(LOG_DIR, f"{title}_{smooth_window_size}.png"), bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smooth", "-s", type=int, default=11, help="smooth window size")
    args = parser.parse_args()

    for env_name in ["HalfCheetah", "Hopper", "Walker2d", "Swimmer", "Ant"]:
        plot_performance_curves(
            title=env_name,
            dir_names=[
                f"{algorithm}_{env_name.lower()}_{seed}"
                for algorithm in ["ppo", "sac", "ddpg"]
                for seed in [42, 729, 1024, 2023, 3500]
            ],
            smooth_window_size=args.smooth,
        )
