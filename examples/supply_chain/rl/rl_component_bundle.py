# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from functools import partial
from typing import Any, Callable, Dict, Optional

import torch

from maro.rl.policy import AbsPolicy
from maro.rl.rl_component.rl_component_bundle import RLComponentBundle
from maro.rl.rollout import AbsEnvSampler
from maro.rl.training import AbsTrainer
from maro.simulator.scenarios.supply_chain import ConsumerUnit, ManufactureUnit
from maro.simulator.scenarios.supply_chain.business_engine import SupplyChainBusinessEngine
from maro.simulator.scenarios.supply_chain.facilities import FacilityInfo
from maro.simulator.scenarios.supply_chain.objects import SupplyChainEntity

from .algorithms.dqn import get_dqn, get_dqn_policy
from .algorithms.ppo import get_ppo, get_ppo_policy
from .algorithms.rule_based import ManufacturerSSPolicy, ConsumerMinMaxPolicy
from .config import ALGO, NUM_CONSUMER_ACTIONS, SHARED_MODEL, env_conf, test_env_conf
from .env_sampler import SCEnvSampler
from .rl_agent_state import STATE_DIM


IS_BASELINE = (ALGO == "EOQ")


def entity2policy(entity: SupplyChainEntity, facility_info_dict: Dict[int, FacilityInfo]) -> Optional[str]:
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
            entity.id: entity
            for entity in helper_business_engine.get_entity_list()
        }

        facility_info_dict: Dict[int, FacilityInfo] = self.env.summary["node_mapping"]["facilities"]

        agent2policy = {
            id_: entity2policy(entity, facility_info_dict)
            for id_, entity in entity_dict.items()
            if entity2policy(entity, facility_info_dict)
        }
        return agent2policy

    def get_policy_creator(self) -> Dict[str, Callable[[], AbsPolicy]]:
        get_policy = (get_dqn_policy if ALGO == "DQN" else get_ppo_policy)
        policy_creator = {
            "consumer_baseline_policy": lambda: ConsumerMinMaxPolicy("consumer_baseline_policy"),
            "manufacturer_policy": lambda: ManufacturerSSPolicy("manufacture_policy"),
            "consumer.policy": partial(get_policy, STATE_DIM, NUM_CONSUMER_ACTIONS, "consumer.policy"),
            "consumer_CA.policy": partial(get_policy, STATE_DIM, NUM_CONSUMER_ACTIONS, "consumer_CA.policy"),
            "consumer_TX.policy": partial(get_policy, STATE_DIM, NUM_CONSUMER_ACTIONS, "consumer_TX.policy"),
            "consumer_WI.policy": partial(get_policy, STATE_DIM, NUM_CONSUMER_ACTIONS, "consumer_WI.policy"),
        }
        return policy_creator

    def get_trainer_creator(self) -> Dict[str, Callable[[], AbsTrainer]]:
        get_trainer = (get_dqn if ALGO=="DQN" else partial(get_ppo, STATE_DIM))
        if SHARED_MODEL:
            trainer_creator = {"consumer": partial(get_trainer, STATE_DIM, "consumer")}
        else:
            trainer_creator = {
                "consumer_CA": partial(get_trainer, "consumer_CA"),
                "consumer_TX": partial(get_trainer, "consumer_TX"),
                "consumer_WI": partial(get_trainer, "consumer_WI"),
            }
        return trainer_creator

    def get_device_mapping(self) -> Dict[str, str]:
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

        if SHARED_MODEL:
            device_mapping = {"consumer.policy": cuda_mapping["shared"]}
        else:
            device_mapping = {
                "consumer_CA.policy": cuda_mapping["CA"],
                "consumer_TX.policy": cuda_mapping["TX"],
                "consumer_WI.policy": cuda_mapping["WI"],
            }
        return device_mapping

    def get_policy_trainer_mapping(self) -> Dict[str, str]:
        if SHARED_MODEL:
            policy_trainer_mapping = {
                "consumer.policy": "consumer"
            }
        else:
            policy_trainer_mapping = {
                "consumer_CA.policy": "consumer_CA",
                "consumer_TX.policy": "consumer_TX",
                "consumer_WI.policy": "consumer_WI",
            }
        return policy_trainer_mapping
