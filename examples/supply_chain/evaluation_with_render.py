# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from os import getcwd

from maro.rl import Actor, LocalLearner

from examples.supply_chain.render_tools import SimulationTracker
from maro.rl.training.message_enums import MsgKey


EPISODE_LEN = 60
N_EPISODE = 1
FACILITY_TYPES = ["productstore"]


class RenderActor(Actor):
    def __init__(
        self, env, policies, agent2policy, group, exploration_dict=None,
        agent2exploration=None, eval_env=None, log_dir=getcwd(), **proxy_kwargs
    ):
        super().__init__(
            env, policies, agent2policy, group, exploration_dict=exploration_dict, agent2exploration=agent2exploration,
            eval_env=eval_env, log_dir=log_dir, **proxy_kwargs
        )
        self._log_dir = log_dir

    def _evaluate(self, msg):
        log_dir = os.path.join(self._log_dir, f"ep_{msg.body[MsgKey.EPISODE_INDEX]}", self._proxy._name)
        tracker = SimulationTracker(
            episode_len=EPISODE_LEN,
            n_episodes=N_EPISODE,
            env=self.eval_env,
            policies=self.policy,
            log_dir=log_dir,
            logger_name=f"SimulationTracker.{self._proxy._name}"
        )
        tracker.run_and_render(facility_types=FACILITY_TYPES)

        return super()._evaluate(msg)


class RenderLocalLearner(LocalLearner):
    def __init__(
        self, env, policies, agent2policy, num_episodes, num_steps=-1, exploration_dict=None, agent2exploration=None,
        eval_schedule=None, eval_env=None, early_stopper=None, log_env_summary=True, log_dir=getcwd()
    ):
        super().__init__(
            env, policies, agent2policy, num_episodes, num_steps=num_steps, exploration_dict=exploration_dict,
            agent2exploration=agent2exploration, eval_schedule=eval_schedule, eval_env=eval_env,
            early_stopper=early_stopper, log_env_summary=log_env_summary, log_dir=log_dir
        )
        self._log_dir = log_dir

    def run(self):
        """Entry point for executing a learning workflow."""
        for ep in range(1, self.num_episodes + 1):
            self._train(ep)
            if ep == self._eval_schedule[self._eval_point_index]:
                self._eval_point_index += 1
                self._evaluate()
                self._run_and_render(ep)
                # early stopping check
                if self.early_stopper:
                    self.early_stopper.push(self.eval_env.summary)
                    if self.early_stopper.stop():
                        return

    def _run_and_render(self, ep: int):
        log_dir = os.path.join(self._log_dir, f"ep_{ep}")
        tracker = SimulationTracker(
            episode_len=EPISODE_LEN,
            n_episodes=N_EPISODE,
            env=self.eval_env,
            policies=self._policy,
            log_dir=log_dir,
            logger_name="SimulationTracker"
        )
        tracker.run_and_render(facility_types=FACILITY_TYPES)

