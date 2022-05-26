# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod
from functools import partial
from typing import Any, Callable, Dict, List, Optional

from maro.rl.policy import AbsPolicy
from maro.rl.rollout import AbsEnvSampler
from maro.rl.training import AbsTrainer
from maro.simulator import Env


class RLComponentBundle(object):
    """Bundle of all necessary components to run a RL job in MARO.

    Users should create their own subclass of `RLComponentBundle` and implement following methods:
    - get_env_config()
    - get_test_env_config()
    - get_env_sampler()
    - get_agent2policy()
    - get_policy_creator()
    - get_trainer_creator()

    Following methods could be overwritten when necessary:
    - get_device_mapping()

    Please refer to the doc string of each method for detailed explanations.
    """

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
        self.policy_trainer_mapping: Optional[Dict[str, str]] = None

        self._policy_cache: Optional[Dict[str, AbsPolicy]] = None

        # Will be created when `env_sampler()` is first called
        self._env_sampler: Optional[AbsEnvSampler] = None

        self._complete_resources()

    ########################################################################################
    # Users MUST implement the following methods                                           #
    ########################################################################################
    @abstractmethod
    def get_env_config(self) -> dict:
        """Return the environment configuration to build the MARO Env for training.

        Returns:
            Environment configuration.
        """
        raise NotImplementedError

    @abstractmethod
    def get_test_env_config(self) -> Optional[dict]:
        """Return the environment configuration to build the MARO Env for testing. If returns `None`, the training
        environment will be reused as testing environment.

        Returns:
            Environment configuration or `None`.
        """
        raise NotImplementedError

    @abstractmethod
    def get_env_sampler(self) -> AbsEnvSampler:
        """Return the environment sampler of the scenario.

        Returns:
            The environment sampler of the scenario.
        """
        raise NotImplementedError

    @abstractmethod
    def get_agent2policy(self) -> Dict[Any, str]:
        """Return agent name to policy name mapping of the RL job. This mapping identifies which policy should
        the agents use. For example: {agent1: policy1, agent2: policy1, agent3: policy2}.

        Returns:
            Agent name to policy name mapping.
        """
        raise NotImplementedError

    @abstractmethod
    def get_policy_creator(self) -> Dict[str, Callable[[], AbsPolicy]]:
        """Return policy creator. Policy creator is a dictionary that contains a group of functions that generate
        policy instances. The key of this dictionary is the policy name, and the value is the function that generate
        the corresponding policy instance. Note that the creation function should not take any parameters.
        """
        raise NotImplementedError

    @abstractmethod
    def get_trainer_creator(self) -> Dict[str, Callable[[], AbsTrainer]]:
        """Return trainer creator. Trainer creator is similar to policy creator, but is used to creator trainers."""
        raise NotImplementedError

    ########################################################################################
    # Users could overwrite the following methods                                          #
    ########################################################################################
    def get_device_mapping(self) -> Dict[str, str]:
        """Return the device mapping that identifying which device to put each policy.

        If user does not overwrite this method, then all policies will be put on CPU by default.
        """
        return {policy_name: "cpu" for policy_name in self.get_policy_creator()}

    def get_policy_trainer_mapping(self) -> Dict[str, str]:
        """Return the policy-trainer mapping which identifying which trainer to train each policy.

        If user does not overwrite this method, then a policy's trainer's name is the first segment of the policy's
        name, seperated by dot. For example, "ppo_1.policy" is trained by "ppo_1".

        Only policies that provided in policy-trainer mapping are considered as trainable polices. Policies that
        not provided in policy-trainer mapping will not be trained since we do not assign a trainer to it.
        """
        return {policy_name: policy_name.split(".")[0] for policy_name in self.policy_creator}

    ########################################################################################
    # Methods invisible to users                                                           #
    ########################################################################################
    @property
    def env_sampler(self) -> AbsEnvSampler:
        if self._env_sampler is None:
            self._env_sampler = self.get_env_sampler()
            self._env_sampler.build(self)
        return self._env_sampler

    def _complete_resources(self) -> None:
        """Generate all attributes by calling user-defined logics. Do necessary checking and transformations."""
        env_config = self.get_env_config()
        test_env_config = self.get_test_env_config()
        self.env = Env(**env_config)
        self.test_env = self.env if test_env_config is None else Env(**test_env_config)

        self.trainer_creator = self.get_trainer_creator()
        self.device_mapping = self.get_device_mapping()

        self.policy_creator = self.get_policy_creator()
        self.agent2policy = self.get_agent2policy()

        self.policy_trainer_mapping = self.get_policy_trainer_mapping()

        required_policies = set(self.agent2policy.values())
        self.policy_creator = {name: self.policy_creator[name] for name in required_policies}
        self.policy_trainer_mapping = {
            name: self.policy_trainer_mapping[name] for name in required_policies if name in self.policy_trainer_mapping
        }
        self.policy_names = list(required_policies)
        assert len(required_policies) == len(self.policy_creator)  # Should have same size after filter

        required_trainers = set(self.policy_trainer_mapping.values())
        self.trainer_creator = {name: self.trainer_creator[name] for name in required_trainers}
        assert len(required_trainers) == len(self.trainer_creator)  # Should have same size after filter

        self.trainable_policy_names = list(self.policy_trainer_mapping.keys())
        self.trainable_policy_creator = {
            policy_name: self.policy_creator[policy_name] for policy_name in self.trainable_policy_names
        }
        self.trainable_agent2policy = {
            agent_name: policy_name
            for agent_name, policy_name in self.agent2policy.items()
            if policy_name in self.trainable_policy_names
        }

    def pre_create_policy_instances(self) -> None:
        """Pre-create policy instances, and return the pre-created policy instances when the external callers
        want to create new policies. This will ensure that each policy will have at most one reusable duplicate.
        Under specific scenarios (for example, simple training & rollout), this will reduce unnecessary overheads.
        """
        old_policy_creator = self.policy_creator
        self._policy_cache: Dict[str, AbsPolicy] = {}
        for policy_name in self.policy_names:
            self._policy_cache[policy_name] = old_policy_creator[policy_name]()

        def _get_policy_instance(policy_name: str) -> AbsPolicy:
            return self._policy_cache[policy_name]

        self.policy_creator = {
            policy_name: partial(_get_policy_instance, policy_name) for policy_name in self.policy_names
        }

        self.trainable_policy_creator = {
            policy_name: self.policy_creator[policy_name] for policy_name in self.trainable_policy_names
        }
