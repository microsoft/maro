# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Optional, Union

import numpy as np

from maro.simulator import Env


def _check_attribute_keys(env: Env, target_type: str, attribute: str) -> None:
    valid_target_types = set(env.summary["node_detail"].keys())
    assert target_type in valid_target_types, f"Target_type {target_type} not in {list(valid_target_types)}!"

    valid_attributes = set(env.summary["node_detail"][target_type]["attributes"].keys())
    assert (
        attribute in valid_attributes
    ), f"Attribute {attribute} not valid for {target_type}. Valid attributes: {list(valid_attributes)}"


def get_attributes(env: Env, target_type: str, attribute: str, tick: Optional[int]) -> np.ndarray:
    _check_attribute_keys(env, target_type, attribute)

    if tick is None:
        num_frame = env.frame_index + 1
        num_nodes = len(env.snapshot_list[target_type])
        return env.snapshot_list[target_type][::attribute].reshape(num_frame, num_nodes)

    else:
        frame_index = env.business_engine.frame_index(tick)
        return env.snapshot_list[target_type][frame_index::attribute].flatten()


def get_list_attributes(
    env: Env,
    target_type: str,
    attribute: str,
    tick: Optional[int],
) -> Union[List[np.ndarray], List[List[np.ndarray]]]:
    _check_attribute_keys(env, target_type, attribute)

    node_indexes = list(range(len(env.snapshot_list[target_type])))

    if tick is None:
        frame_indexes = list(range(env.frame_index + 1))
        return [
            [env.snapshot_list[target_type][frame_index:index:attribute].flatten() for index in node_indexes]
            for frame_index in frame_indexes
        ]

    else:
        frame_index = env.business_engine.frame_index(tick)
        return [env.snapshot_list[target_type][frame_index:index:attribute].flatten() for index in node_indexes]
