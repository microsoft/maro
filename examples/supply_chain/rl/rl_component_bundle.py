from functools import partial
from typing import Any, Callable, Dict, Optional

from maro.rl.policy import AbsPolicy
from maro.rl.rl_component.rl_component_bundle import RLComponentBundle
from maro.rl.rollout import AbsEnvSampler
from maro.rl.training import AbsTrainer
from maro.simulator.scenarios.supply_chain import ConsumerUnit, ManufactureUnit
from maro.simulator.scenarios.supply_chain.business_engine import SupplyChainBusinessEngine
from maro.simulator.scenarios.supply_chain.facilities import FacilityInfo
from maro.simulator.scenarios.supply_chain.objects import SupplyChainEntity

from .algorithms.ppo import get_ppo, get_ppo_policy
from .algorithms.rule_based import ConsumerMinMaxPolicy
from .config import NUM_CONSUMER_ACTIONS, env_conf, use_or_policy
from .env_sampler import SCEnvSampler
from .rl_agent_state import STATE_DIM


def entity2policy(entity: SupplyChainEntity, facility_info_dict: Dict[int, FacilityInfo]) -> Optional[str]:
    if issubclass(entity.class_type, ManufactureUnit):
        return None
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
        return "consumer_policy" if use_or_policy else "ppo.policy"
    return None


class SupplyChainBundle(RLComponentBundle):
    def get_env_config(self) -> dict:
        return env_conf

    def get_test_env_config(self) -> Optional[dict]:
        return None

    def get_env_sampler(self) -> AbsEnvSampler:
        return SCEnvSampler(self.env, self.test_env)

    def get_agent2policy(self) -> Dict[Any, str]:
        helper_business_engine = self.env.business_engine
        assert isinstance(helper_business_engine, SupplyChainBusinessEngine)

        entity_dict: Dict[Any, SupplyChainEntity] = {
            entity.id: entity
            for entity in helper_business_engine.get_entity_list()
        }

        return {
            id_: entity2policy(entity, self.env.summary["node_mapping"]["facilities"])
            for id_, entity in entity_dict.items()
            if entity2policy(entity, self.env.summary["node_mapping"]["facilities"])
        }

    def get_policy_creator(self) -> Dict[str, Callable[[], AbsPolicy]]:
        if use_or_policy:
            return {"consumer_policy": lambda: ConsumerMinMaxPolicy("consumer_policy")}
        else:
            return {"ppo.policy": partial(get_ppo_policy, STATE_DIM, NUM_CONSUMER_ACTIONS, "ppo.policy")}

    def get_trainer_creator(self) -> Dict[str, Callable[[], AbsTrainer]]:
        return {} if use_or_policy else {"ppo": partial(get_ppo, STATE_DIM, "ppo")}

    def get_policy_trainer_mapping(self) -> Dict[str, str]:
        return {} if use_or_policy else {"ppo.policy": "ppo"}
