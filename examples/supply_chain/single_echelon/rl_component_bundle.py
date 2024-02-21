# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from functools import partial
from typing import Any, Callable, Dict, Optional

import torch

from maro.rl.policy import AbsPolicy
from maro.rl.rl_component.rl_component_bundle import RLComponentBundle
from maro.rl.rollout import AbsEnvSampler
from maro.rl.training import AbsTrainer
from maro.simulator.scenarios.supply_chain import ConsumerUnit
from maro.simulator.scenarios.supply_chain.business_engine import SupplyChainBusinessEngine
from maro.simulator.scenarios.supply_chain.facilities import FacilityInfo
from maro.simulator.scenarios.supply_chain.objects import SupplyChainEntity

from examples.supply_chain.rl.algorithms.dqn import get_dqn, get_dqn_policy
from examples.supply_chain.rl.algorithms.ppo import get_ppo, get_ppo_policy
from examples.supply_chain.rl.algorithms.rule_based import ConsumerMinMaxPolicy
from .config import ALGO, NUM_CONSUMER_ACTIONS, env_conf, test_env_conf
from .env_sampler import SCEnvSampler
from examples.supply_chain.rl.rl_agent_state import STATE_DIM

IS_BASELINE = ALGO == "EOQ"


def entity2policy(entity: SupplyChainEntity, facility_info_dict: Dict[int, FacilityInfo]) -> Optional[str]:
    if issubclass(entity.class_type, ConsumerUnit):
        facility_name = facility_info_dict[entity.facility_id].name
        if not IS_BASELINE:
            # Could be independent for different facilities
            return "consumer.policy"
        else:
            return "consumer_baseline_policy"

    return None


class SupplyChainBundle(RLComponentBundle):
    def get_env_config(self) -> dict:
        return env_conf

    def get_test_env_config(self) -> Optional[dict]:
        return test_env_conf

    def get_env_sampler(self) -> AbsEnvSampler:
        return SCEnvSampler(self.env, self.test_env, reward_eval_delay=None)

    def get_agent2policy(self) -> Dict[Any, str]:
        helper_business_engine = self.env.business_engine
        assert isinstance(helper_business_engine, SupplyChainBusinessEngine)

        entity_dict: Dict[Any, SupplyChainEntity] = {
            entity.id: entity for entity in helper_business_engine.get_entity_list()
        }

        facility_info_dict: Dict[int, FacilityInfo] = self.env.summary["node_mapping"]["facilities"]

        agent2policy = {
            id_: entity2policy(entity, facility_info_dict)
            for id_, entity in entity_dict.items()
            if entity2policy(entity, facility_info_dict)
        }
        return agent2policy

    def get_policy_creator(self) -> Dict[str, Callable[[], AbsPolicy]]:
        get_policy = get_dqn_policy if ALGO == "DQN" else get_ppo_policy
        policy_creator = {
            "consumer_baseline_policy": lambda: ConsumerMinMaxPolicy("consumer_baseline_policy"),
            "consumer.policy": partial(get_policy, STATE_DIM, NUM_CONSUMER_ACTIONS, "consumer.policy"),
        }
        return policy_creator

    def get_trainer_creator(self) -> Dict[str, Callable[[], AbsTrainer]]:
        get_trainer = get_dqn if ALGO == "DQN" else partial(get_ppo, STATE_DIM)
        trainer_creator = {"consumer": partial(get_trainer, STATE_DIM, "consumer")}
        return trainer_creator

    def get_device_mapping(self) -> Dict[str, str]:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        device_mapping = {"consumer.policy": device}
        return device_mapping

    def get_policy_trainer_mapping(self) -> Dict[str, str]:
        policy_trainer_mapping = {"consumer.policy": "consumer"}
        return policy_trainer_mapping
