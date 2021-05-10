# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import heapq
import time
from collections import defaultdict, namedtuple
from os import getcwd
from typing import Callable, Dict, List, Tuple, Union

from maro.rl.env_wrapper import AbsEnvWrapper
from maro.rl.exploration import AbsExploration
from maro.rl.policy import AbsCorePolicy, AbsPolicy
from maro.utils import Logger

from .policy_update_schedule import MultiPolicyUpdateSchedule


class LocalLearner(object):
    """Learner class for distributed training.

    Args:
        policy (MultiAgentPolicy): Learning agents.
        
    """
    def __init__(
        self,
        policy_dict: Dict[str, AbsPolicy],
        agent2policy: Dict[str, str],
        env: AbsEnvWrapper,
        num_episodes: int,
        policy_update_schedule: MultiPolicyUpdateSchedule,
        exploration_dict: Dict[str, AbsExploration] = None,
        agent2exploration: Dict[str, str] = None,
        experience_update_interval: int = -1,
        eval_env: AbsEnvWrapper = None,
        eval_schedule: Union[int, List[int]] = None,
        log_env_metrics: bool = True,
        log_total_reward: bool = True,
        log_dir: str = getcwd(),
        **end_of_episode_kwargs
    ):
        self._logger = Logger("LEARNER", dump_folder=log_dir)

        # mappings between agents and policies
        self.policy_dict = policy_dict
        self.agent2policy = agent2policy
        self.policy = {agent_id: policy_dict[policy_id] for agent_id, policy_id in self.agent2policy.items()}
        self.agent_groups_by_policy = defaultdict(list)
        for agent_id, policy_id in agent2policy.items():
            self.agent_groups_by_policy[policy_id].append(agent_id)

        for policy_id, agent_ids in self.agent_groups_by_policy.items():
            self.agent_groups_by_policy[policy_id] = tuple(agent_ids)

        self.agent_groups_by_policy = defaultdict(list)
        for agent_id, policy_id in agent2policy.items():
            self.agent_groups_by_policy[policy_id].append(agent_id)

        for policy_id, agent_ids in self.agent_groups_by_policy.items():
            self.agent_groups_by_policy[policy_id] = tuple(agent_ids)

        # mappings between exploration schemes and agents
        self.exploration_dict = exploration_dict
        if exploration_dict:
            self.agent2exploration = agent2exploration
            self.exploration = {
                agent_id: self.exploration_dict[exploration_id]
                for agent_id, exploration_id in self.agent2exploration.items()
            }
            self.exploration_enabled = True
            self.agent_groups_by_exploration = defaultdict(list)
            for agent_id, exploration_id in agent2exploration.items():
                self.agent_groups_by_exploration[exploration_id].append(agent_id)

            for exploration_id, agent_ids in self.agent_groups_by_exploration.items():
                self.agent_groups_by_exploration[exploration_id] = tuple(agent_ids)

        self.env = env
        self.num_episodes = num_episodes
        self.policy_update_schedule = policy_update_schedule       
        self.eval_env = eval_env if eval_env else self.env
        self.experience_update_interval = experience_update_interval

        # evaluation schedule
        if eval_schedule is None:
            eval_schedule = []
        elif isinstance(eval_schedule, int):
            num_eval_schedule = num_episodes // eval_schedule
            eval_schedule = [eval_schedule * i for i in range(1, num_eval_schedule + 1)]

        self.eval_schedule = eval_schedule
        self.eval_schedule.sort()
        if not self.eval_schedule or num_episodes != self.eval_schedule[-1]:
            self.eval_schedule.append(num_episodes)

        self._logger.info(f"Policy will be evaluated at the end of episodes {self.eval_schedule}")
        self._eval_point_index = 0

        self.end_of_episode_kwargs = end_of_episode_kwargs
        self._log_env_metrics = log_env_metrics
        self._log_total_reward = log_total_reward

    def run(self):
        for ep in range(1, self.num_episodes + 1):
            self._train(ep)

            policy_ids = self.policy_update_schedule.pop_episode(ep)
            if policy_ids == ["*"]:
                policy_ids = list(self.policy_dict.keys())
            for policy_id in policy_ids:
                self.policy_dict[policy_id].update()

            if policy_ids:
                self._logger.info(f"Updated policies {policy_ids} at the end of episode {ep}")

            if ep == self.eval_schedule[self._eval_point_index]:
                self._eval_point_index += 1
                self._evaluate(self._eval_point_index)

            self.end_of_episode(ep, **self.end_of_episode_kwargs)

    def _train(self, ep: int):
        t0 = time.time()
        learning_time = 0
        num_experiences_collected = 0

        self._logger.info(f"Training episode {ep}")
        if self.exploration_dict:
            exploration_params = {
                agent_ids: self.exploration_dict[exploration_id].parameters
                for exploration_id, agent_ids in self.agent_groups_by_exploration.items()
            }
            self._logger.debug(f"Exploration parameters: {exploration_params}")

        self.env.save_replay = True
        self.env.reset()
        self.env.start()  # get initial state
        while self.env.state:
            if self.exploration_dict:      
                action = {
                    id_:
                        self.exploration[id_](self.policy[id_].choose_action(st))
                        if id_ in self.exploration else self.policy[id_].choose_action(st)
                    for id_, st in self.env.state.items()
                }
            else:
                action = {id_: self.policy[id_].choose_action(st) for id_, st in self.env.state.items()}

            self.env.step(action)
            step_index = self.env.step_index

            # experience collection
            if not self.env.state or step_index % self.experience_update_interval == 0:
                exp_by_agent = self.env.get_experiences()
                for agent_id, exp in exp_by_agent.items():
                    if isinstance(self.policy[agent_id], AbsCorePolicy):
                        self.policy[agent_id].experience_manager.put(exp)
                num_experiences_collected += sum(len(exp) for exp in exp_by_agent.values())

            # policy update
            tl0 = time.time()
            policy_ids = self.policy_update_schedule.pop_step(ep, step_index)
            if policy_ids == ["*"]:
                policy_ids = list(self.policy_dict.keys())
            for policy_id in policy_ids:
                self.policy_dict[policy_id].update()

            if policy_ids:
                self._logger.info(f"Updated policies {policy_ids} after step {step_index}")
            learning_time += time.time() - tl0

        # update the exploration parameters
        if self.exploration_dict:
            for exploration in self.exploration_dict.values():
                exploration.step()

        # performance details
        if self._log_env_metrics:
            self._logger.info(f"ep {ep}: {self.env.metrics}")
        if self._log_total_reward:
            self._logger.info(f"ep {ep} total reward received: {self.env.total_reward}")
        self._logger.debug(
            f"ep {ep} summary - "
            f"running time: {time.time() - t0}"
            f"env steps: {self.env.step_index}"    
            f"learning time: {learning_time}"
            f"experiences collected: {num_experiences_collected}"
        )

    def _evaluate(self, ep: int):
        self._logger.info("Evaluating...")
        self.eval_env.save_replay = False
        self.eval_env.reset()
        self.eval_env.start()  # get initial state
        while self.eval_env.state:
            action = {id_: self.policy[id_].choose_action(st) for id_, st in self.eval_env.state.items()}
            self.eval_env.step(action)

        if self._log_env_metrics:
            self._logger.info(f"eval ep {ep}: {self.eval_env.metrics}")
        if not self.eval_env.state:
            self._logger.info(f"total reward: {self.eval_env.total_reward}")

    def end_of_episode(self, ep: int, **kwargs):
        pass
