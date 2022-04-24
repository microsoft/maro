from abc import abstractmethod
from functools import partial
from typing import Any, Callable, Dict, List, Optional

from maro.rl.policy import AbsPolicy
from maro.rl.rollout import AbsEnvSampler
from maro.rl.training import AbsTrainer
from maro.rl.utils import extract_trainer_name
from maro.simulator import Env


class RLComponentBundle(object):
    def __init__(self) -> None:
        super(RLComponentBundle, self).__init__()

        self.trainer_creator: Optional[Dict[str, Callable[[], AbsTrainer]]] = None

        self.agent2policy: Optional[Dict[Any, str]] = None
        self.trainable_agent2policy: Optional[Dict[Any, str]] = None
        self.policy_creator: Optional[Dict[str, Callable[[], AbsPolicy]]] = None
        self.policy_names: Optional[List[str]] = None
        self.trainable_policy_creator: Optional[Dict[str, Callable[[], AbsPolicy]]] = None
        self.trainable_policy_names: Optional[List[str]] = None

        self.device_mapping: Optional[Dict[str, str]] = None

        self._policy_cache: Optional[Dict[str, AbsPolicy]] = None

    ########################################################################################
    @abstractmethod
    def get_env_config(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def get_test_env_config(self) -> Optional[dict]:
        raise NotImplementedError

    @abstractmethod
    def get_env_sampler(self) -> AbsEnvSampler:
        raise NotImplementedError

    @abstractmethod
    def get_agent2policy(self) -> Dict[Any, str]:
        raise NotImplementedError

    @abstractmethod
    def get_policy_creator(self) -> Dict[str, Callable[[], AbsPolicy]]:
        raise NotImplementedError

    @abstractmethod
    def get_trainer_creator(self) -> Dict[str, Callable[[], AbsTrainer]]:
        raise NotImplementedError

    def get_device_mapping(self) -> Dict[str, str]:
        return {policy_name: "cpu" for policy_name in self.get_policy_creator()}

    def post_collect(self, info_list: list, ep: int, segment: int) -> None:
        pass

    def post_evaluate(self, info_list: list, ep: int) -> None:
        pass

    ########################################################################################
    def complete_resources(self) -> None:
        env_config = self.get_env_config()
        test_env_config = self.get_test_env_config()
        self.env = Env(**env_config)
        self.test_env = self.env if test_env_config is None else Env(**test_env_config)

        self.trainer_creator = self.get_trainer_creator()
        self.device_mapping = self.get_device_mapping()

        self.policy_creator = self.get_policy_creator()
        self.policy_names = list(self.policy_creator.keys())
        self.agent2policy = self.get_agent2policy()

        self.trainable_policy_names = [
            policy_name for policy_name in self.policy_names
            if extract_trainer_name(policy_name) in self.trainer_creator
        ]
        self.trainable_policy_creator = {
            self.policy_creator[policy_name] for policy_name in self.trainable_policy_names
        }
        self.trainable_agent2policy = {
            agent_name: policy_name
            for agent_name, policy_name in self.agent2policy.items()
            if policy_name in self.trainable_policy_names
        }

    def pre_create_policy_instances(self) -> None:
        old_policy_creator = self.policy_creator
        self._policy_cache: Dict[str, AbsPolicy] = {}
        for policy_name in self.policy_names:
            self._policy_cache[policy_name] = old_policy_creator[policy_name]()

        def _get_policy_instance(policy_name: str) -> AbsPolicy:
            return self._policy_cache[policy_name]

        self.policy_creator = {
            policy_name: partial(_get_policy_instance, policy_name)
            for policy_name in self.policy_names
        }

        self.trainable_policy_creator = {
            policy_name: self.policy_creator[policy_name]
            for policy_name in self.trainable_policy_names
        }
