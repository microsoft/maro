# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from functools import partial
from typing import Any, Dict, Optional

from maro.simulator import Env
from maro.simulator.scenarios.supply_chain import ConsumerUnit, ManufactureUnit
from maro.simulator.scenarios.supply_chain.business_engine import SupplyChainBusinessEngine
from maro.simulator.scenarios.supply_chain.facilities import FacilityInfo
from maro.simulator.scenarios.supply_chain.objects import SupplyChainEntity

from .rl_agent_state import STATE_DIM
from .algorithms.ppo import get_policy, get_ppo
<<<<<<< HEAD
from .algorithms.rule_based import DummyPolicy, ManufacturerBaselinePolicy, ConsumerEOQPolicy
from .config import NUM_CONSUMER_ACTIONS, env_conf, ALGO, SHARED_MODEL
=======
from .algorithms.rule_based import DummyPolicy, ConsumerMinMaxPolicy
from .config import NUM_CONSUMER_ACTIONS, env_conf
>>>>>>> origin/Jinyu/sc_refinement


# Create an env to get entity list and env summary
env = Env(**env_conf)

facility_info_dict: Dict[int, FacilityInfo] = env.summary["node_mapping"]["facilities"]

helper_business_engine = env.business_engine
assert isinstance(helper_business_engine, SupplyChainBusinessEngine)

entity_dict: Dict[Any, SupplyChainEntity] = {
    entity.id: entity
    for entity in helper_business_engine.get_entity_list()
}


<<<<<<< HEAD
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
=======
def entity2policy(entity: SupplyChainEntity) -> Optional[str]:
    if issubclass(entity.class_type, ManufactureUnit):
>>>>>>> origin/Jinyu/sc_refinement
        return "manufacturer_policy"

    elif issubclass(entity.class_type, ConsumerUnit):
        facility_name = facility_info_dict[entity.facility_id].name
<<<<<<< HEAD
        if baseline:
            return "consumer_eoq_policy"
        if facility_name.startswith("CA_") or facility_name.startswith("TX_") or facility_name.startswith("WI_"):
            return ("consumer.policy" if SHARED_MODEL else f"consumer.{facility_name}.policy")
        else:
            return "consumer_eoq_policy"
    else:
        raise TypeError(f"Unrecognized entity class type: {entity.class_type}")

policy_creator = {
    "consumer.policy": partial(get_policy, STATE_DIM, NUM_CONSUMER_ACTIONS),
    "consumer_eoq_policy": lambda name: ConsumerEOQPolicy(name),
    "manufacturer_policy": lambda name: ManufacturerBaselinePolicy(name),
    "facility_policy": lambda name: DummyPolicy(name),
    "product_policy": lambda name: DummyPolicy(name),
    "seller_policy": lambda name: DummyPolicy(name),
}

trainable_policies = ["consumer.policy"]
=======
        if "Plant" in facility_name:
            # Return the policy name if needed
            pass
        elif "Warehouse" in facility_name:
            # Return the policy name if needed
            pass
        elif "Store" in facility_name:
            # Return the policy name if needed
            pass
        return "ppo.policy"
        # return "consumer_policy"

    return None


policy_creator = {
    "ppo.policy": partial(get_policy, STATE_DIM, NUM_CONSUMER_ACTIONS),
    "manufacturer_policy": lambda name: DummyPolicy(name),
    "consumer_policy": lambda name: ConsumerMinMaxPolicy(name),
}

agent2policy = {
    id_: entity2policy(entity)
    for id_, entity in entity_dict.items()
    if entity2policy(entity)
}
>>>>>>> origin/Jinyu/sc_refinement

if not SHARED_MODEL:
    for entity_id, entity in facility_info_dict.items():
        if issubclass(entity.class_type, ConsumerUnit):
            facility_name = facility_info_dict[entity.facility_id].name
            policy_key = f"consumer.{facility_name}.policy"
            if policy_key not in policy_creator:
                trainable_policies.append(policy_key)
                policy_creator[policy_key] = partial(get_policy, STATE_DIM, NUM_CONSUMER_ACTIONS)


if ALGO != "EOQ":
    agent2policy = {
        id_: entity2policy(entity, False) for id_, entity in entity_dict.items()
    }
else:
    # baseline policies
    agent2policy = {
        id_: entity2policy(entity, True) for id_, entity in entity_dict.items()
    }


device_mapping = {"consumer.policy": "cuda"}