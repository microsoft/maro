# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from functools import partial
from typing import Any, Dict

import torch

from maro.simulator import Env
from maro.simulator.scenarios.supply_chain import ConsumerUnit, ManufactureUnit
from maro.simulator.scenarios.supply_chain.business_engine import SupplyChainBusinessEngine
from maro.simulator.scenarios.supply_chain.facilities import FacilityInfo
from maro.simulator.scenarios.supply_chain.objects import SupplyChainEntity

from .algorithms.ppo import get_policy as get_ppo_policy
from .algorithms.ppo import get_ppo
from .algorithms.dqn import get_policy as get_dqn_policy
from .algorithms.dqn import get_dqn

from .algorithms.rule_based import ManufacturerSSPolicy, ConsumerMinMaxPolicy
from .config import env_conf, ALGO, NUM_CONSUMER_ACTIONS, SHARED_MODEL
from .rl_agent_state import STATE_DIM


IS_BASELINE = (ALGO == "EOQ")

# Create an env to get entity list and env summary
env = Env(**env_conf)

facility_info_dict: Dict[int, FacilityInfo] = env.summary["node_mapping"]["facilities"]

helper_business_engine = env.business_engine
assert isinstance(helper_business_engine, SupplyChainBusinessEngine)

entity_dict: Dict[Any, SupplyChainEntity] = {
    entity.id: entity
    for entity in helper_business_engine.get_entity_list()
}

# Define the rule of policy mapping
def entity2policy(entity: SupplyChainEntity) -> str:
    if issubclass(entity.class_type, ManufactureUnit):
        return "manufacturer_policy"

    elif issubclass(entity.class_type, ConsumerUnit):
        facility_name = facility_info_dict[entity.facility_id].name
        if not IS_BASELINE and any([
            facility_name.startswith("CA_"),
            facility_name.startswith("TX_"),
            facility_name.startswith("WI_"),
        ]):
            if SHARED_MODEL:
                return "consumer.policy"
            else:
                return f"consumer_{facility_name[:2]}.policy"

        else:
            return "consumer_baseline_policy"

    return None

agent2policy = {
    id_: entity2policy(entity)
    for id_, entity in entity_dict.items()
    if entity2policy(entity)
}

get_policy = (get_dqn_policy if ALGO == "DQN" else get_ppo_policy)
policy_creator = {
    "consumer_baseline_policy": lambda name: ConsumerMinMaxPolicy(name),
    "manufacturer_policy": lambda name: ManufacturerSSPolicy(name),
    "consumer.policy": partial(get_policy, STATE_DIM, NUM_CONSUMER_ACTIONS),
    "consumer_CA.policy": partial(get_policy, STATE_DIM, NUM_CONSUMER_ACTIONS),
    "consumer_TX.policy": partial(get_policy, STATE_DIM, NUM_CONSUMER_ACTIONS),
    "consumer_WI.policy": partial(get_policy, STATE_DIM, NUM_CONSUMER_ACTIONS),
}

# Define basic device mapping
cuda_mapping = {
    "CA": "cpu",
    "TX": "cpu",
    "WI": "cpu",
    "shared": "cpu",
}
if torch.cuda.is_available():
    gpu_cnts = torch.cuda.device_count()
    cuda_mapping = {
        "CA": f"cuda:{0 % gpu_cnts}",
        "TX": f"cuda:{1 % gpu_cnts}",
        "WI": f"cuda:{2 % gpu_cnts}",
        "shared": "cuda:0",
    }

get_trainer = (get_dqn if ALGO=="DQN" else partial(get_ppo, STATE_DIM))
if SHARED_MODEL:
    trainable_policies = ["consumer.policy"]
    trainer_creator = {"consumer": get_trainer}
    device_mapping = {"consumer.policy": cuda_mapping["shared"]}
else:
    trainable_policies = [
        "consumer_CA.policy",
        "consumer_TX.policy",
        "consumer_WI.policy",
    ]
    trainer_creator = {
        "consumer_CA": get_trainer,
        "consumer_TX": get_trainer,
        "consumer_WI": get_trainer,
    }
    device_mapping = {
        "consumer_CA.policy": cuda_mapping["CA"],
        "consumer_TX.policy": cuda_mapping["TX"],
        "consumer_WI.policy": cuda_mapping["WI"],
    }
