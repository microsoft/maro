# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.simulator import Env

env_conf = {
    "scenario": "vm_scheduling",
    "topology": "azure.2019.10k",
    "start_tick": 0,
    "durations": 300,  # 8638
    "snapshot_resolution": 1,
}

num_pms = Env(**env_conf).business_engine.pm_amount
pm_window_size = 1
num_features = 2 * num_pms * pm_window_size + 4
state_dim = num_features + num_pms + 1

pm_attributes = ["cpu_cores_capacity", "memory_capacity", "cpu_cores_allocated", "memory_allocated"]
# vm_attributes = ["cpu_cores_requirement", "memory_requirement", "lifetime", "remain_time", "total_income"]


reward_shaping_conf = {
    "alpha": 0.0,
    "beta": 1.0,
}
seed = 666

test_env_conf = {
    "scenario": "vm_scheduling",
    "topology": "azure.2019.10k.oversubscription",
    "start_tick": 0,
    "durations": 300,
    "snapshot_resolution": 1,
}
test_reward_shaping_conf = {
    "alpha": 0.0,
    "beta": 1.0,
}

test_seed = 1024

algorithm = "ac"  # "dqn" or "ac"
