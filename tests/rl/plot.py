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
    x, y = df["n_steps"], df["val/avg_reward"]
    x = np.cumsum(x)
    mask = ~np.isnan(y)
    x, y = x[mask], y[mask]
    return x, y


def get_on_policy_data(log_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    file_path = os.path.join(log_dir, "metrics_full.csv")
    df = pd.read_csv(file_path)
    x, y = df["n_steps"], df["avg_reward"]
    x = np.cumsum(x)
    return x, y


def plot_performance_curves(title: str, dir_names: List[str], smooth_window_size: int) -> None:
    for name in dir_names:
        log_dir = os.path.join(LOG_DIR, name)
        if not os.path.exists(log_dir):
            continue

        if "ppo" in name:
            algorithm = "ppo"
            func = get_on_policy_data
        elif "sac" in name:
            algorithm = "sac"
            func = get_off_policy_data
        else:
            raise "unknown algorithm name"

        x, y = func(log_dir)
        y = smooth(y, smooth_window_size)
        plt.plot(x, y, label=algorithm, color=color_map[algorithm])

    plt.legend()
    plt.title(title)
    plt.xlabel("Total Env Interactions")
    plt.ylabel(f"Average Trajectory Return (moving average with window size = {smooth_window_size})")
    plt.savefig(os.path.join(LOG_DIR, f"{title}_{smooth_window_size}.png"))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smooth", "-s", type=int, default=11, help="smooth window size")
    args = parser.parse_args()

    for env_name in ["HalfCheetah", "Hopper", "Walker2d", "Swimmer", "Ant"]:
        plot_performance_curves(
            title=env_name,
            dir_names=[f"{algorithm}_{env_name.lower()}" for algorithm in ["ppo", "sac"]],
            smooth_window_size=args.smooth,
        )
