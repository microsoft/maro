# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np


def _show_info(rewards: list, tag: str) -> None:
    print(
        f"[{tag}] Total N-steps = {sum([len(e) for e in rewards])}, "
        f"N segments = {len(rewards)}, "
        f"Average reward = {np.mean([sum(e) for e in rewards]):.4f}, "
        f"Max reward = {np.max([sum(e) for e in rewards]):.4f}, "
        f"Min reward = {np.min([sum(e) for e in rewards]):.4f}, "
        f"Average N-steps = {np.mean([len(e) for e in rewards]):.1f}\n",
    )


def post_collect(info_list: list, ep: int, segment: int) -> None:
    rewards = [list(e["env_metric"]["reward_record"].values()) for e in info_list]
    _show_info(rewards, "Collect")


def post_evaluate(info_list: list, ep: int) -> None:
    rewards = [list(e["env_metric"]["reward_record"].values()) for e in info_list]
    _show_info(rewards, "Evaluate")
