# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys
import time
import numpy as np
from typing import Union
import matplotlib.pyplot as plt
from collections import defaultdict

import torch

from maro.rl import (
    AbsEnvWrapper, AbsPolicy, AbsExploration, AbsEarlyStopper
)
from maro.utils import DummyLogger, Logger

plt.switch_backend('agg')


class VMLearner:
    def __init__(
        self,
        env: AbsEnvWrapper,
        policy: AbsPolicy,
        auxiliary_policy: object,

        num_episodes: int,
        num_steps: int = -1,
        exploration: AbsExploration = None,
        eval_schedule: int = None,
        eval_env: AbsEnvWrapper = None,
        early_stopper: AbsEarlyStopper = None,

        auxiliary_prob: float = None,

        simulation_logger: Logger = DummyLogger(),
        eval_simulation_logger: Logger = DummyLogger(),
        rl_logger: Logger = DummyLogger(),
        eval_rl_logger: Logger = DummyLogger(),
        model_path: str = "",
        picture_path: str = ""
    ):
        self.env = env
        self.eval_env = eval_env if eval_env else self.env

        self._policy = policy
        self._auxiliary_policy = auxiliary_policy

        self.num_episodes = num_episodes
        self._num_steps = num_steps if num_steps > 0 else float("inf")

        self._exploration = exploration

        num_eval_schedule = num_episodes // eval_schedule
        eval_schedule = [eval_schedule * i for i in range(1, num_eval_schedule + 1)]

        self._eval_schedule = eval_schedule
        self._eval_schedule.sort()
        if not self._eval_schedule or num_episodes != self._eval_schedule[-1]:
            self._eval_schedule.append(num_episodes)
        self._eval_point_index = 0
        self._eval_ep = 0

        self.early_stopper = early_stopper

        self._auxiliary = False
        self._auxiliary_prob = auxiliary_prob

        self._simulation_logger = simulation_logger
        self._eval_simulation_logger = eval_simulation_logger
        self._rl_logger = rl_logger
        self._eval_rl_logger = eval_rl_logger

        self._model_path = model_path
        self._picture_path = picture_path

    def run(self):
        self._evaluate()
        for ep in range(1, self.num_episodes + 1):
            self._train(ep)
            if ep == self._eval_schedule[self._eval_point_index]:
                self._eval_point_index += 1
                self._evaluate()
                # early stopping check
                if self.early_stopper:
                    self.early_stopper.push(self.eval_env.summary)
                    if self.early_stopper.stop():
                        return

    def _train(self, ep: int):
        """Collect simulation data for training."""
        t0 = time.time()
        learning_time = 0
        num_experiences_collected = 0

        if self._exploration:
            exploration_params = self._exploration.parameters
            self._rl_logger.debug(f"Exploration parameters: {exploration_params}")

        # whether to use the rule-based agent
        if np.random.random() > (1 - self._auxiliary_prob):
            self._auxiliary = True
        else:
            self._auxiliary = False

        self.env.reset()
        self.env.start()  # get initial state
        segment = 0
        while self.env.state is not None:
            segment += 1
            exp = self._collect(ep, segment)
            self._policy.on_experiences(exp)

        # update the exploration parameters if an episode is finished
        if self._exploration:
            self._exploration.step()

        # performance details
        self._simulation_logger.info(f"ep {ep}: {self.env.summary}")

        self._rl_logger.debug(
            f"ep {ep} summary - "
            f"running time: {time.time() - t0} "
            f"env steps: {self.env.step_index} "
            f"learning time: {learning_time} "
            f"experiences collected: {num_experiences_collected}"
        )

    def _evaluate(self):
        """Evaluate the performance of the model in the train environment and test environment."""
        self._eval_ep += 1

        self.env.reset()
        self.env.start()  # get initial state
        while self.env.state is not None:
            action, value = self._policy.choose_action(self.env.state, self.env.legal_pm, False)
            self.env.step(action)

        # performance details
        self._eval_simulation_logger.info(f"evaluation ep {self._eval_ep}: {self.env.summary}")

        self.eval_env.reset()
        self.eval_env.start()  # get initial state
        info = defaultdict(list)
        actions = []
        while self.eval_env.state is not None:
            action, value = self._policy.choose_action(self.eval_env.state, self.eval_env.legal_pm, False)
            legal_pm = self.eval_env.legal_pm.copy()
            info[self.eval_env.event.vm_cpu_cores_requirement].append([value, legal_pm])
            actions.append(action)
            self.eval_env.step(action)

        # performance details
        self._eval_simulation_logger.info(f"evaluation ep {self._eval_ep}: {self.eval_env.summary}")

        # draw info (q_values or action_prob) and action sequence
        self.draw(info, actions, self._eval_ep)

        self.dump_models(f"{self._model_path}/epoch_{self._eval_ep}")

    def _collect(self, ep, segment):
        start_step_index = self.env.step_index + 1
        steps_to_go = self._num_steps
        while self.env.state is not None and steps_to_go:
            if self._auxiliary:
                action = self._auxiliary_policy.choose_action(self.env.event)
            else:
                if self._exploration:
                    action = self._exploration(
                        self._policy.choose_action(self.env.state, self.env.legal_pm), self.env.legal_pm
                    )
                else:
                    action = self._policy.choose_action(self.env.state, self.env.legal_pm)
            self.env.step(action)
            steps_to_go -= 1

        self._simulation_logger.info(
            f"Roll-out finished for ep {ep}, segment {segment}"
            f"(steps {start_step_index} - {self.env.step_index})"
        )

        return self.env.get_experiences()

    def draw(self, values, actions, current_iter):
        self.draw_action_sequence(actions, current_iter)
        self.draw_with_legal_action(values, current_iter)
        self.draw_without_legal_action(values, current_iter)

    def draw_action_sequence(self, actions, current_iter):
        fig = plt.figure(figsize=(40, 32))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(actions)
        fig.savefig(f"{self._picture_path}/action_sequence_{current_iter}")
        plt.cla()
        plt.close("all")

    def draw_with_legal_action(self, values, current_iter):
        fig = plt.figure(figsize=(40, 32))
        for idx, key in enumerate(values.keys()):
            ax = fig.add_subplot(len(values.keys()), 1, idx + 1)
            for i in range(len(values[key])):
                if i == 0:
                    ax.plot(values[key][i][0] * values[key][i][1], label=str(key))
                    ax.legend()
                else:
                    ax.plot(values[key][i][0] * values[key][i][1])

        fig.savefig(f"{self._picture_path}/values_with_legal_action_{current_iter}")

        plt.cla()
        plt.close("all")

    def draw_without_legal_action(self, values, current_iter):
        fig = plt.figure(figsize=(40, 32))

        for idx, key in enumerate(values.keys()):
            ax = fig.add_subplot(len(values.keys()), 1, idx + 1)
            for i in range(len(values[key])):
                if i == 0:
                    ax.plot(values[key][i][0], label=str(key))
                    ax.legend()
                else:
                    ax.plot(values[key][i][0])

        fig.savefig(f"{self._picture_path}/values_without_legal_action_{current_iter}")

        plt.cla()
        plt.close("all")

    def dump_models(self, dir_path: str):
        os.makedirs(dir_path, exist_ok=True)
        torch.save(self._policy.get_state(), os.path.join(dir_path, "model.pt"))
