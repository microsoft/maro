# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from functools import partial

from maro.simulator.scenarios.supply_chain import ConsumerUnit, ManufactureUnit, ProductUnit, SellerUnit
from maro.simulator.scenarios.supply_chain.world import SupplyChainEntity
from .algorithms.rule_based import or_policy_creator
from .config import rl_algorithm, NUM_CONSUMER_ACTIONS
from .env_helper import entity_dict
from .state_template import STATE_DIM


def get_policy_name(entity: SupplyChainEntity) -> str:
    if entity.skus is None:
        return "facility_policy"
    elif issubclass(entity.class_type, ManufactureUnit):
        return "manufacturer_policy"
    elif issubclass(entity.class_type, ProductUnit):
        return "product_policy"
    elif issubclass(entity.class_type, ConsumerUnit):
        return "consumer_policy"
    elif issubclass(entity.class_type, SellerUnit):
        return "seller_policy"
    raise TypeError(f"Unrecognized entity class type: {entity.class_type}")


agent2policy = {id_: get_policy_name(entity) for id_, entity in entity_dict.items()}

if rl_algorithm == "dqn":
    from .algorithms.dqn import get_dqn, get_policy
    policy_creator = {
        policy_name: partial(get_policy, STATE_DIM, NUM_CONSUMER_ACTIONS)
        for policy_name in agent2policy.values()
    }
    trainer_creator = {
        agent_name.split(".")[0]: get_dqn for agent_name in agent2policy.values()
    }
elif rl_algorithm == "ppo":
    from .algorithms.ppo import get_policy, get_ppo
    policy_creator = {
        policy_name: partial(get_policy, STATE_DIM, NUM_CONSUMER_ACTIONS)
        for policy_name in agent2policy.values()
    }
    trainer_creator = {
        policy_name.split(".")[0]: partial(get_ppo, STATE_DIM) for policy_name in agent2policy.values()
    }
elif rl_algorithm == 'or':
    policy_creator = or_policy_creator
    trainer_creator = {}
else:
    raise ValueError(f"Unsupported algorithm: {rl_algorithm}")
