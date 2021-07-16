# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
from collections import defaultdict
from os import getcwd
from typing import Dict, List, Union

from maro.rl.env_wrapper import AbsEnvWrapper
from maro.rl.policy import AbsPolicy, AbsCorePolicy
from maro.utils import Logger

from .actor_manager import ActorManager
from .policy_update_schedule import EpisodeBasedSchedule, MultiPolicyUpdateSchedule, StepBasedSchedule


class DistributedLearner(object):
    """Learner class for distributed training.

    Args:
        policy (MultiAgentPolicy): Learning agents.

    """
    def __init__(
        self,
        policy_dict: Dict[str, AbsPolicy],
        agent_to_policy: Dict[str, str],
        num_episodes: int,
        policy_update_schedule: Union[EpisodeBasedSchedule, StepBasedSchedule, dict],
        actor_manager: ActorManager,
        experience_update_interval: int = -1,
        eval_env: AbsEnvWrapper = None,
        eval_schedule: Union[int, List[int]] = None,
        num_eval_actors: int = 1,
        discard_stale_experiences: bool = True,
        log_env_metrics: bool = True,
        log_dir: str = getcwd(),
        **end_of_episode_kwargs
    ):
        self._logger = Logger("LEARNER", dump_folder=log_dir)
        self.policy_dict = policy_dict
        self.agent_to_policy = agent_to_policy
        self.policy = {agent_id: policy_dict[policy_id] for agent_id, policy_id in self.agent_to_policy.items()}
        self.agent_groups_by_policy = defaultdict(list)
        for agent_id, policy_id in agent_to_policy.items():
            self.agent_groups_by_policy[policy_id].append(agent_id)

        for policy_id, agent_ids in self.agent_groups_by_policy.items():
            self.agent_groups_by_policy[policy_id] = tuple(agent_ids)

        self.num_episodes = num_episodes
        self.policy_update_schedule = MultiPolicyUpdateSchedule(policy_update_schedule)
        if isinstance(policy_update_schedule, dict):
            self._updatable_policy_ids = list(policy_update_schedule.keys())
        else:
            self._updatable_policy_ids = [
                policy_id for policy_id, policy in self.policy_dict.items() if hasattr(policy, "update")
            ]
        self._updated_policy_ids = self._updatable_policy_ids

        self.experience_update_interval = experience_update_interval

        # evaluation
        self.eval_env = eval_env
        self.num_eval_actors = num_eval_actors

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

        self.actor_manager = actor_manager
        self.discard_stale_experiences = discard_stale_experiences

        self.end_of_episode_kwargs = end_of_episode_kwargs
        self._log_env_metrics = log_env_metrics

        self._action_dict = {}

    def run(self):
        for ep in range(1, self.num_episodes + 1):
            self._logger.info(f"Learner episode: {ep}")
            self._train(ep)

            policy_ids = self.policy_update_schedule.pop_episode(ep)
            if policy_ids == ["*"]:
                policy_ids = self._updatable_policy_ids

            for policy_id in policy_ids:
                self.policy_dict[policy_id].update()

            if policy_ids:
                self._logger.info(f"Updated policies {policy_ids} at the end of episode {ep}")

            self._updated_policy_ids = policy_ids
            self.end_of_episode(ep, **self.end_of_episode_kwargs)

            if ep == self.eval_schedule[self._eval_point_index]:
                self._eval_point_index += 1
                self._evaluate(self._eval_point_index)

        if self.actor_manager:
            self.actor_manager.exit()

    def _train(self, ep: int):
        t0 = time.time()
        learning_time = 0
        env_steps = 0
        num_experiences_collected = 0
        num_actor_finishes, segment_index = 0, 1

        while num_actor_finishes < self.actor_manager.required_finishes:
            # parallel experience collection
            for exp_by_agent, done in self.actor_manager.collect(
                ep, segment_index, self.experience_update_interval,
                policy_dict={policy_id: self.policy_dict[policy_id].get_state() for policy_id in self._updated_policy_ids},
                discard_stale_experiences=self.discard_stale_experiences
            ):
                for agent_id, exp in exp_by_agent.items():
                    if isinstance(self.policy[agent_id], AbsCorePolicy):
                        self.policy[agent_id].experience_manager.put(exp)
                        if agent_id not in self._action_dict.keys():
                            self._action_dict[agent_id] = {}
                        for a in exp.actions:
                            self._action_dict[agent_id][a] = self._action_dict[agent_id].get(a, 0) + 1

                env_steps += self.experience_update_interval
                num_experiences_collected += sum(len(exp) for exp in exp_by_agent.values())
                num_actor_finishes += done

            # policy update
            tl0 = time.time()
            policy_ids = self.policy_update_schedule.pop_step(ep, segment_index)
            if policy_ids == ["*"]:
                policy_ids = self._updatable_policy_ids
            for policy_id in policy_ids:
                self.policy_dict[policy_id].update()

            if policy_ids:
                self._logger.info(f"Updated policies {policy_ids} after segment {segment_index}")

            self._updated_policy_ids = policy_ids
            learning_time += time.time() - tl0

            segment_index += 1

        # performance details
        self._logger.debug(
            f"ep {ep} summary - "
            f"running time: {time.time() - t0} - "
            f"env steps: {env_steps} - "
            f"learning time: {learning_time} - "
            f"experiences collected: {num_experiences_collected} - ",
            f"action dist: {self._action_dict} - "
        )

    def _evaluate(self, ep: int):
        self._logger.info("Evaluating...")
        if self.eval_env:
            self.eval_env.save_replay = False
            self.eval_env.reset()
            self.eval_env.start()  # get initial state
            while self.eval_env.state:
                action = {id_: self.policy[id_].choose_action(st) for id_, st in self.eval_env.state.items()}
                self.eval_env.step(action)

            if not self.eval_env.state:
                self._logger.info(f"total reward: {self.eval_env.total_reward}")

            if self._log_env_metrics:
                self._logger.info(f"eval ep {ep}: {self.eval_env.metrics}")
        else:
            policy_state = {
                policy_id: self.policy_dict[policy_id].get_state() for policy_id in self._updatable_policy_ids
            }
            self.actor_manager.evaluate(ep, policy_state, self.num_eval_actors)

    def end_of_episode(self, ep: int, **kwargs):
        pass
