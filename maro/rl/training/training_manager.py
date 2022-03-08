# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import os
from itertools import chain
from typing import Callable, Dict, Iterable, List, Tuple

from maro.rl.policy import RLPolicy
from maro.rl.rollout import ExpElement
from maro.rl.training.abs_algorithm import AbsAlgorithm
from maro.utils import LoggerV2

from .utils import extract_algo_inst_name, get_training_state_path


class TrainingManager(object):
    """
    Training manager. Manage and schedule all algorithm instances to train policies.

    Args:
        policy_creator (Dict[str, Callable[[str], RLPolicy]]): Dict of functions to create policies.
        algo_inst_creator (Dict[str, Callable[[str], AbsAlgorithm]]): Dict of functions to create algorithm instances.
        agent2policy (Dict[str, str]): Agent name to policy name mapping.
        proxy_address (Tuple[str, int], default=None): Address of the training proxy. If it is not None,
            it is registered to all algorithm instances, which in turn create `RemoteOps` for distributed training.
    """

    def __init__(
        self,
        policy_creator: Dict[str, Callable[[str], RLPolicy]],
        algo_inst_creator: Dict[str, Callable[[str], AbsAlgorithm]],
        agent2policy: Dict[str, str],  # {agent_name: policy_name}
        proxy_address: Tuple[str, int] = None,
        logger: LoggerV2 = None,
    ) -> None:
        super(TrainingManager, self).__init__()

        self._algo_inst_dict: Dict[str, AbsAlgorithm] = {}
        self._agent2policy = agent2policy
        self._proxy_address = proxy_address
        for algo_inst_name, func in algo_inst_creator.items():
            algo_inst = func(algo_inst_name)
            if self._proxy_address:
                algo_inst.set_proxy_address(self._proxy_address)
            algo_inst.register_agent2policy(self._agent2policy)
            algo_inst.register_policy_creator(policy_creator)
            algo_inst.register_logger(logger)
            algo_inst.build()  # `build()` must be called after `register_policy_creator()`
            self._algo_inst_dict[algo_inst_name] = algo_inst

        self._agent_to_algo_inst = {
            agent_name: extract_algo_inst_name(policy_name)
            for agent_name, policy_name in self._agent2policy.items()
        }

    def train(self) -> None:
        if self._proxy_address:
            async def train_step() -> Iterable:
                return await asyncio.gather(*[
                    algo_inst.train_step_as_task() for algo_inst in self._algo_inst_dict.values()
                ])

            asyncio.run(train_step())
        else:
            for algo_inst in self._algo_inst_dict.values():
                algo_inst.train_step()

    def get_policy_state(self) -> Dict[str, Dict[str, object]]:
        """Get policies' states.

        Returns:
            A double-deck dict with format: {algo_inst_name: {policy_name: policy_state}}
        """
        return dict(chain(*[algo_inst.get_policy_state().items() for algo_inst in self._algo_inst_dict.values()]))

    def record_experiences(self, experiences: List[List[ExpElement]]) -> None:
        """Record experiences collected from external modules (for example, EnvSampler).

        Args:
            experiences (List[ExpElement]): List of experiences. Each ExpElement stores the complete information for a
                tick. Please refers to the definition of ExpElement for detailed explanation of ExpElement.
        """
        for env_idx, env_experience in enumerate(experiences):
            for exp_element in env_experience:  # Dispatch experiences to algorithm instances tick by tick.
                exp_dict = exp_element.split_contents(self._agent_to_algo_inst)
                for algo_inst_name, exp_elem in exp_dict.items():
                    algo_inst = self._algo_inst_dict[algo_inst_name]
                    algo_inst.record(env_idx, exp_elem)

    def load(self, path: str) -> List[str]:
        loaded = []
        for algo_inst_name, algo_inst in self._algo_inst_dict.items():
            pth = get_training_state_path(path, algo_inst_name)
            if os.path.isfile(pth):
                algo_inst.load(pth)
                loaded.append(algo_inst_name)

        return loaded

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        for algo_inst_name, algo_inst in self._algo_inst_dict.items():
            algo_inst.save(get_training_state_path(path, algo_inst_name))

    def exit(self) -> None:
        if self._proxy_address:
            async def exit_all() -> Iterable:
                return await asyncio.gather(*[algo_inst.exit() for algo_inst in self._algo_inst_dict.values()])

            asyncio.run(exit_all())
