# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from os import getcwd
from typing import Union

from maro.communication import Proxy
from maro.rl.policy import MultiAgentPolicy 
from maro.rl.env_wrapper import AbsEnvWrapper
from maro.utils import Logger

from .message_enums import MsgKey, MsgTag


class Actor(object):
    """On-demand roll-out executor.

    Args:
        env (AbsEnvWrapper): An ``AbsEnvWrapper`` instance that wraps an ``Env`` instance with scenario-specific
            processing logic and stores transitions during roll-outs in a replay memory.
        policy (MultiAgentPolicy): Agent that interacts with the environment.
        group (str): Identifier of the group to which the actor belongs. It must be the same group name
            assigned to the learner (and decision clients, if any).
        proxy_options (dict): Keyword parameters for the internal ``Proxy`` instance. See ``Proxy`` class
            for details. Defaults to None.
    """
    def __init__(
        self,
        env: AbsEnvWrapper,
        policy: MultiAgentPolicy,
        group: str,
        eval_env: AbsEnvWrapper = None,
        proxy_options: dict = None,
        log_dir: str = getcwd()
    ):
        self.env = env
        self.eval_env = eval_env if eval_env else self.env
        self.policy = policy
        if proxy_options is None:
            proxy_options = {}
        self._proxy = Proxy(group, "actor", {"actor_manager": 1}, **proxy_options)
        self._logger = Logger(self._proxy.name, dump_folder=log_dir)

    def run(self):
        for msg in self._proxy.receive():
            if msg.tag == MsgTag.EXIT:
                self._logger.info("Exiting...")
                break

            if msg.tag == MsgTag.COLLECT:
                self.policy.train_mode()
                episode_index, segment_index = msg.body[MsgKey.EPISODE_INDEX], msg.body[MsgKey.SEGMENT_INDEX]
                if self.env.state is None:
                    self._logger.info(f"Training with exploration parameters: {self.policy.exploration_params}")
                    self.env.reset()
                    self.env.start()  # get initial state

                starting_step_index = self.env.step_index + 1
                self.policy.load_state(msg.body[MsgKey.POLICY])
                steps_to_go = float("inf") if msg.body[MsgKey.NUM_STEPS] == -1 else msg.body[MsgKey.NUM_STEPS]
                while self.env.state and steps_to_go > 0:
                    action = self.policy.choose_action(self.env.state)
                    self.env.step(action)
                    steps_to_go -= 1

                self._logger.info(
                    f"Roll-out finished for ep {episode_index}, segment {segment_index}"
                    f"(steps {starting_step_index} - {self.env.step_index})"
                )
                return_info = {
                    MsgKey.ENV_END: not self.env.state,
                    MsgKey.EPISODE_INDEX: episode_index,
                    MsgKey.SEGMENT_INDEX: segment_index,
                    MsgKey.EXPERIENCES: self.env.get_experiences(),
                    MsgKey.NUM_STEPS: self.env.step_index - starting_step_index + 1
                }

                if msg.body[MsgKey.RETURN_ENV_METRICS]:
                    return_info[MsgKey.METRICS] = self.env.metrics
                if not self.env.state:
                    self.policy.exploration_step()
                    return_info[MsgKey.TOTAL_REWARD] = self.env.total_reward
                self._proxy.reply(msg, tag=MsgTag.COLLECT_DONE, body=return_info)
            elif msg.tag == MsgTag.EVAL:
                ep = msg.body[MsgKey.EPISODE_INDEX]
                self._logger.info(f"Evaluation episode {ep}")
                self.policy.eval_mode()
                self.eval_env.reset()
                self.eval_env.start()  # get initial state
                self.policy.load_state(msg.body[MsgKey.POLICY])
                while self.eval_env.state:
                    self.eval_env.step(self.policy.choose_action(self.eval_env.state))

                return_info = {
                    MsgKey.METRICS: self.env.metrics,
                    MsgKey.TOTAL_REWARD: self.eval_env.total_reward,
                    MsgKey.EPISODE_INDEX: msg.body[MsgKey.EPISODE_INDEX]  
                }
                self._proxy.reply(msg, tag=MsgTag.EVAL_DONE, body=return_info)
