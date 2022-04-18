# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from functools import partial
from typing import Any, Dict

from maro.simulator import Env
from maro.simulator.scenarios.supply_chain import ConsumerUnit, ManufactureUnit, ProductUnit, SellerUnit
from maro.simulator.scenarios.supply_chain.business_engine import SupplyChainBusinessEngine
from maro.simulator.scenarios.supply_chain.facilities import FacilityInfo
from maro.simulator.scenarios.supply_chain.objects import SupplyChainEntity

from .agent_state import STATE_DIM
from .algorithms.ppo import get_policy, get_ppo
from .algorithms.rule_based import DummyPolicy, ManufacturerBaselinePolicy, ConsumerEOQPolicy
from .config import NUM_CONSUMER_ACTIONS, env_conf, ALGO


# Create an env to get entity list and env summary
env = Env(**env_conf)

facility_info_dict: Dict[int, FacilityInfo] = env.summary["node_mapping"]["facilities"]

helper_business_engine = env.business_engine
assert isinstance(helper_business_engine, SupplyChainBusinessEngine)

entity_dict: Dict[Any, SupplyChainEntity] = {
    entity.id: entity
    for entity in helper_business_engine.get_entity_list()
}


# Create an env to get entity list and env summary
env = Env(**env_conf)

facility_info_dict: Dict[int, FacilityInfo] = env.summary["node_mapping"]["facilities"]

helper_business_engine = env.business_engine
assert isinstance(helper_business_engine, SupplyChainBusinessEngine)

entity_dict: Dict[Any, SupplyChainEntity] = {
    entity.id: entity
    for entity in helper_business_engine.get_entity_list()
}


if ALGO == "PPO":
    from .algorithms.ppo import get_policy, get_ppo
    trainer_creator = {
        "consumer": partial(get_ppo, STATE_DIM),
    }
else:
    from .algorithms.dqn import get_dqn, get_policy
    trainer_creator = {
        "consumer": get_dqn,
    }



def entity2policy(entity: SupplyChainEntity, baseline) -> str:
    if entity.skus is None:
        return "facility_policy"
    elif issubclass(entity.class_type, ManufactureUnit):
        return "manufacturer.policy"
    elif issubclass(entity.class_type, ProductUnit):
        return "product_policy"
    elif issubclass(entity.class_type, SellerUnit):
        return "seller_policy"
    elif issubclass(entity.class_type, ConsumerUnit):
        facility_name = facility_info_dict[entity.facility_id].name
        if "Plant" in facility_name:
            # Return the policy name if needed
            pass
        elif "Warehouse" in facility_name:
            # Return the policy name if needed
            pass
        elif "Store" in facility_name:
            # Return the policy name if needed
            pass
        return ("consumer.eoq_policy" if baseline else "consumer.policy")
    raise TypeError(f"Unrecognized entity class type: {entity.class_type}")


policy_creator = {
    "consumer.policy": partial(get_policy, STATE_DIM, NUM_CONSUMER_ACTIONS),
    "consumer.eoq_policy": lambda name: ConsumerEOQPolicy(name),
    "manufacturer.policy": lambda name: ManufacturerBaselinePolicy(name),
    "facility_policy": lambda name: DummyPolicy(name),
    "product_policy": lambda name: DummyPolicy(name),
    "seller_policy": lambda name: DummyPolicy(name),
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

trainable_policies = ["consumer.policy"]

device_mapping = {"consumer.policy": "cuda"}