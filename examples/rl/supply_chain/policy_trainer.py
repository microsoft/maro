# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from functools import partial

from maro.simulator.scenarios.supply_chain import ConsumerUnit, ManufactureUnit, ProductUnit
from maro.simulator.scenarios.supply_chain.world import SupplyChainEntity
from .config import rl_algorithm, NUM_CONSUMER_ACTIONS
from .env_helper import entity_dict
from .state_template import STATE_DIM


def get_policy_name(entity: SupplyChainEntity) -> str:
    if entity.is_facility:
        return "facility_policy"
    elif entity.class_type == ManufactureUnit:
        return "manufacturer_policy"
    elif entity.class_type == ProductUnit:
        return "product_policy"
    elif entity.class_type == ConsumerUnit:
        return f"dqn_{entity.id}.policy"


agent2policy = {}
for id_, entity in entity_dict.items():
    if issubclass(entity.class_type, ManufactureUnit):
        agent2policy[id_] = f"dqn_manufacture_{entity.id}.policy"
    elif issubclass(entity.class_type, ConsumerUnit):
        agent2policy[id_] = f"dqn_consumer_{entity.id}.policy"

if rl_algorithm == "dqn":
    from .algorithms.dqn import get_dqn, get_policy
    policy_creator = {
        agent_name: partial(get_policy, STATE_DIM, NUM_CONSUMER_ACTIONS) for agent_name in agent2policy.values()
    }
    trainer_creator = {
        agent_name.split(".")[0]: get_dqn for agent_name in agent2policy.values()
    }
else:
    raise ValueError(f"Unsupported algorithm: {rl_algorithm}")
