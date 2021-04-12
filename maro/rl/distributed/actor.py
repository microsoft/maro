# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from os import getcwd
from typing import Union

from maro.communication import Message, Proxy
from maro.rl.agent import AbsAgent, MultiAgentWrapper
from maro.rl.training import AbsEnvWrapper
from maro.utils import Logger

from .message_enums import MsgKey, MsgTag


class Actor(object):
    """On-demand roll-out executor.

    Args:
        env (AbsEnvWrapper): An ``AbsEnvWrapper`` instance that wraps an ``Env`` instance with scenario-specific
            processing logic and stores transitions during roll-outs in a replay memory.
        agent (Union[AbsAgent, MultiAgentWrapper]): Agent that interacts with the environment.
        group (str): Identifier of the group to which the actor belongs. It must be the same group name
            assigned to the learner (and decision clients, if any).
        proxy_options (dict): Keyword parameters for the internal ``Proxy`` instance. See ``Proxy`` class
            for details. Defaults to None.
    """
    def __init__(
        self,
        env: AbsEnvWrapper,
        agent: Union[AbsAgent, MultiAgentWrapper],
        group: str,
        proxy_options: dict = None,
        pull_experiences_with_copy: bool = False,
        log_dir: str = getcwd()
    ):
        self.env = env
        self.agent = MultiAgentWrapper(agent) if isinstance(agent, AbsAgent) else agent
        self._pull_experiences_with_copy = pull_experiences_with_copy
        if proxy_options is None:
            proxy_options = {}
        self._proxy = Proxy(group, "actor", {"actor_manager": 1}, **proxy_options)
        self._logger = Logger(self._proxy.name, dump_folder=log_dir)

    def run(self):
        for msg in self._proxy.receive():
            if msg.tag == MsgTag.EXIT:
                self._logger.info("Exiting...")
                break
            if msg.tag == MsgTag.ROLLOUT:
                rollout_index, segment_index = msg.body[MsgKey.ROLLOUT_INDEX], msg.body[MsgKey.SEGMENT_INDEX]
                if self.env.state is None:
                    self.env.reset()
                    # Load exploration parameters
                    if MsgKey.EXPLORATION_PARAMS in msg.body:
                        self.agent.set_exploration_params(msg.body[MsgKey.EXPLORATION_PARAMS])
                    self.env.start(rollout_index=rollout_index)  # get initial state

                starting_step_index = self.env.step_index
                self.agent.load_model(msg.body[MsgKey.MODEL])
                steps_to_go = float("inf") if msg.body[MsgKey.NUM_STEPS] == -1 else msg.body[MsgKey.NUM_STEPS]
                while self.env.state and steps_to_go > 0:
                    action = self.agent.choose_action(self.env.state)
                    self.env.step(action)
                    steps_to_go -= 1 

                self._logger.info(
                    f"Roll-out finished for ep {rollout_index}, segment {segment_index}"
                    f"(steps {starting_step_index} - {self.env.step_index})"
                )
                experiences, num_exp = self.env.pull_experiences(copy=self._pull_experiences_with_copy)
                self._proxy.reply(
                    msg, 
                    tag=MsgTag.ROLLOUT_DONE,
                    body={
                        MsgKey.ENV_END: not self.env.state,
                        MsgKey.ROLLOUT_INDEX: rollout_index,
                        MsgKey.SEGMENT_INDEX: segment_index,
                        MsgKey.METRICS: self.env.metrics,
                        MsgKey.EXPERIENCES: experiences,
                        MsgKey.NUM_STEPS: self.env.step_index - starting_step_index,
                        MsgKey.NUM_EXPERIENCES: num_exp
                    }
                )
