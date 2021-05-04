# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
from collections import defaultdict
from os import getcwd
from typing import Callable, Dict, List, Union

from maro.communication import Message, Proxy, SessionType
from maro.rl.env_wrapper import AbsEnvWrapper
from maro.rl.policy import AbsCorePolicy, MultiAgentPolicy
from maro.utils import Logger

from .actor_manager import ActorManager
from .message_enums import MsgTag, MsgKey


class Learner(object):
    """Learner class for distributed training.

    Args:
        policy (MultiAgentPolicy): Learning agents.
        
    """
    def __init__(
        self,
        policy: MultiAgentPolicy,
        env: AbsEnvWrapper,
        num_episodes: int,
        eval_env: AbsEnvWrapper = None,
        actor_manager: ActorManager = None,
        num_eval_actors: int = 1,
        policy_update_interval: int = -1,
        eval_points: Union[int, List[int]] = None,
        required_actor_finishes: str = None,
        discard_stale_experiences: bool = True,
        log_env_metrics: bool = True,
        log_dir: str = getcwd(),
        end_of_training_kwargs: dict = None
    ):
        if env is None and actor_manager is None:
            raise Exception("env and actor_manager cannot both be None")

        self._logger = Logger("LEARNER", dump_folder=log_dir)
        self.policy = policy
        self.env = env
        self.num_episodes = num_episodes
        self.eval_env = eval_env if eval_env else self.env
        self.policy_update_interval = policy_update_interval

        if actor_manager:
            self._logger.info(
                "Found actor manager. Roll-outs will be performed by remote actors. Local env will not be used."
            )
        self.actor_manager = actor_manager

        self.num_eval_actors = num_eval_actors

        # evaluation points
        if eval_points is None:
            eval_points = []
        elif isinstance(eval_points, int):
            num_eval_points = num_episodes // eval_points
            eval_points = [eval_points * i for i in range(1, num_eval_points + 1)]

        self.eval_points = eval_points
        self.eval_points.sort()
        if not self.eval_points or num_episodes != self.eval_points[-1]:
            self.eval_points.append(num_episodes)

        self._logger.info(f"Policy will be evaluated at the end of episodes {self.eval_points}")
        self._eval_point_index = 0

        # distributed roll-out
        self.actor_manager = actor_manager
        if self.actor_manager:
            if required_actor_finishes and required_actor_finishes > self.actor_manager.num_actors:
                raise ValueError("required_actor_finishes cannot exceed the number of available actors")

            if required_actor_finishes is None:
                required_actor_finishes = self.actor_manager.num_actors
                self._logger.info(f"Required number of actor finishes is set to {required_actor_finishes}")

            self.required_actor_finishes = required_actor_finishes
            self.discard_stale_experiences = discard_stale_experiences

        self.end_of_training_kwargs = end_of_training_kwargs if end_of_training_kwargs else {} 
        self._log_env_metrics = log_env_metrics
        self._total_learning_time = 0
        self._total_env_steps = 0
        self._total_experiences_collected = 0

    def run(self):
        for ep in range(1, self.num_episodes + 1):
            self.train(ep)
            if ep == self.eval_points[self._eval_point_index]:
                self._eval_point_index += 1
                self.evaluate(self._eval_point_index)

        if self.actor_manager:
            self.actor_manager.exit()

    def train(self, ep: int):
        # local mode
        if not self.actor_manager:
            self._train_local(ep)
        else:
            self._train_remote(ep)

    def evaluate(self, ep: int):
        self._logger.info("Evaluating...")
        if self.eval_env:
            self.policy.eval_mode()
            self.eval_env.save_replay = False
            self.eval_env.reset()
            self.eval_env.start()  # get initial state
            while self.eval_env.state:
                action = self.policy.choose_action(self.eval_env.state)
                self.eval_env.step(action)

            if not self.eval_env.state:
                self._logger.info(f"total reward: {self.eval_env.total_reward}")

            if self._log_env_metrics:
                self._logger.info(f"eval ep {ep}: {self.eval_env.metrics}")
        else:
            self.actor_manager.evaluate(ep, self.policy.state(), self.num_eval_actors)

    def _train_local(self, ep: int):
        t0 = time.time()
        segement_index = 1
        self._logger.info(f"Training episode {ep}") 
        self._logger.debug(f"exploration parameters: {self.policy.exploration_params}")
        self.policy.train_mode()
        self.env.save_replay = True
        self.env.reset()
        self.env.start()  # get initial state
        while self.env.state:
            action = self.policy.choose_action(self.env.state)
            self.env.step(action)
            if (
                not self.env.state or
                self.policy_update_interval != -1 and self.env.step_index % self.policy_update_interval == 0
            ):
                exp_by_agent = self.env.get_experiences()
                tl0 = time.time()
                self.policy.store_experiences(exp_by_agent)
                updated_policy_ids = self.policy.update()
                self._logger.info(f"updated policies {updated_policy_ids}")
                self.end_of_training(ep, segement_index, **self.end_of_training_kwargs)
                self._total_learning_time += time.time() - tl0
                self._total_env_steps += self.policy_update_interval
                self._total_experiences_collected += sum(len(exp) for exp in exp_by_agent.values())
                self._logger.debug(f"total running time: {time.time() - t0}")
                self._logger.debug(f"total learning time: {self._total_learning_time}")
                self._logger.debug(f"total env steps: {self._total_env_steps}")
                self._logger.debug(f"total experiences collected: {self._total_experiences_collected}")
                if not self.env.state:
                    self._logger.info(f"total reward: {self.env.total_reward}")

                segement_index += 1

        self.policy.exploration_step()
        if self._log_env_metrics:
            self._logger.info(f"ep {ep}: {self.env.metrics}")

    def _train_remote(self, ep: int):
        t0 = time.time()
        updated_policy_ids, num_actor_finishes, segment_index = list(self.policy.policy_dict.keys()), 0, 0
        while num_actor_finishes < self.required_actor_finishes:
            for exp, done in self.actor_manager.collect(
                ep, segment_index, self.policy_update_interval,
                policy_dict=self.policy.state(),
                required_actor_finishes=self.required_actor_finishes,
                discard_stale_experiences=self.discard_stale_experiences
            ):
                tl0 = time.time()
                self.policy.store_experiences(exp)
                updated_policy_ids = self.policy.update()
                self._logger.info(f"updated policies {updated_policy_ids}")
                self.end_of_training(ep, segment_index, **self.end_of_training_kwargs)
                num_actor_finishes += done
                self._total_learning_time += time.time() - tl0
                self._logger.debug(f"running time: {time.time() - t0}")
                self._logger.debug(f"learning time: {self._total_learning_time}")
                self._logger.debug(f"env steps: {self.actor_manager.total_env_steps}")
                self._logger.debug(f"experiences collected: {self.actor_manager.total_experiences_collected}")

            segment_index += 1

    def end_of_training(self, ep: int, segment: int, **kwargs):
        pass
