# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from functools import partial
from typing import Any, Dict, Optional

from maro.simulator import Env
from maro.simulator.scenarios.supply_chain import ConsumerUnit, ManufactureUnit
from maro.simulator.scenarios.supply_chain.business_engine import SupplyChainBusinessEngine
from maro.simulator.scenarios.supply_chain.facilities import FacilityInfo
from maro.simulator.scenarios.supply_chain.objects import SupplyChainEntity
from maro.simulator.scenarios.supply_chain.units.product import ProductUnit
from maro.simulator.scenarios.supply_chain.units.seller import SellerUnit

from .algorithms.ppo import get_policy as get_ppo_policy
from .algorithms.ppo import get_ppo
from .algorithms.dqn import get_policy as get_dqn_policy
from .algorithms.dqn import get_dqn


from .rl_agent_state import STATE_DIM
from .algorithms.rule_based import DummyPolicy, ManufacturerSSPolicy, ConsumerMinMaxPolicy
from .config import NUM_CONSUMER_ACTIONS, env_conf, ALGO, SHARED_MODEL

import torch

gpu_available = torch.cuda.is_available()
gpu_cnts = torch.cuda.device_count()



# Create an env to get entity list and env summary
env = Env(**env_conf)

facility_info_dict: Dict[int, FacilityInfo] = env.summary["node_mapping"]["facilities"]

helper_business_engine = env.business_engine
assert isinstance(helper_business_engine, SupplyChainBusinessEngine)

entity_dict: Dict[Any, SupplyChainEntity] = {
    entity.id: entity
    for entity in helper_business_engine.get_entity_list()
}


get_policy = (get_dqn_policy if ALGO == "DQN" else get_ppo_policy)


def entity2policy(entity: SupplyChainEntity, baseline) -> str:
    if entity.skus is None:
        return "facility_policy"
    elif issubclass(entity.class_type, ProductUnit):
        return "product_policy"
    elif issubclass(entity.class_type, ManufactureUnit):
        return "manufacturer_policy"
    elif issubclass(entity.class_type, SellerUnit):
        return "seller_policy"
    elif issubclass(entity.class_type, ConsumerUnit):
        facility_name = facility_info_dict[entity.facility_id].name
        if baseline:
            return "consumer_eoq_policy"
        if facility_name.startswith("CA_") or facility_name.startswith("TX_") or facility_name.startswith("WI_"):
            return ("consumer.policy" if SHARED_MODEL else f"consumer_{facility_name[:2]}.policy")
        else:
            return "consumer_eoq_policy"
    else:
        raise TypeError(f"Unrecognized entity class type: {entity.class_type}")

policy_creator = {
    "consumer_eoq_policy": lambda name: ConsumerMinMaxPolicy(name),
    "manufacturer_policy": lambda name: ManufacturerSSPolicy(name),
    "facility_policy": lambda name: DummyPolicy(name),
    "product_policy": lambda name: DummyPolicy(name),
    "seller_policy": lambda name: DummyPolicy(name),
}



cuda_mapping = {"CA": "cpu", "TX": "cpu", "WI": "cpu"}
if gpu_available:
    cuda_mapping = {"CA": f"cuda:{0%gpu_cnts}", "TX": f"cuda:{1%gpu_cnts}", "WI": f"cuda:{2%gpu_cnts}"}
if not SHARED_MODEL:
    trainer_creator = {}
    trainable_policies = []
    device_mapping = {}
    for entity_id, entity in entity_dict.items():
        if issubclass(entity.class_type, ConsumerUnit):
            facility_name = facility_info_dict[entity.facility_id].name
            if facility_name.startswith("CA_") or facility_name.startswith("TX_") or facility_name.startswith("WI_"):
                policy_key = f"consumer_{facility_name[:2]}.policy"
                if policy_key not in policy_creator.keys():
                    trainable_policies.append(policy_key)
                    device_mapping[policy_key] = cuda_mapping[facility_name[:2]]
                    policy_creator[policy_key] = partial(get_policy, STATE_DIM, NUM_CONSUMER_ACTIONS)
                trainer_key = f"consumer_{facility_name[:2]}"
                if trainer_key not in trainer_creator.keys():
                    trainer_creator[trainer_key] = (get_dqn if ALGO=="DQN" else partial(get_ppo, STATE_DIM))
else:
    trainable_policies = ["consumer.policy"]
    device_mapping = {"consumer.policy": ("cuda:0" if gpu_available else "cpu")}
    policy_creator["consumer.policy"] =  partial(get_policy, STATE_DIM, NUM_CONSUMER_ACTIONS)
    if ALGO == "PPO":
        trainer_creator = {
            "consumer": partial(get_ppo, STATE_DIM),
        }
    else:
        trainer_creator = {
            "consumer": get_dqn,
        }


if ALGO != "EOQ":
    agent2policy = {
        id_: entity2policy(entity, False) for id_, entity in entity_dict.items()
    }
else:
    # baseline policies
    agent2policy = {
        id_: entity2policy(entity, True) for id_, entity in entity_dict.items()
    }

