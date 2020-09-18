# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Union

from maro.simulator import Env
from .abstract_actor import AbstractActor, RolloutMode
from maro.rl.agent.agent_manager import AgentManager
from maro.simulator import Env


class SimpleActor(AbstractActor):
    """
    A simple actor class that implements typical roll-out logic
    """
    def __init__(self, env: Union[dict, Env], inference_agents: AgentManager):
        assert isinstance(inference_agents, AgentManager), \
            "SimpleActor only accepts type AgentManager for parameter inference_agents"
        super().__int__(env, inference_agents)

    def roll_out(self, mode, models=None, epsilon_dict=None, seed: int = None):
        if mode == RolloutMode.EXIT:
            return None, None

        env = self._env if isinstance(self._env, Env) else self._env[mode]
        if seed is not None:
            env.set_seed(seed)

        # assign epsilons
        if epsilon_dict is not None:
            self._inference_agents.explorer.epsilon = epsilon_dict

        # load models
        if models is not None:
            self._inference_agents.load_models(models)

        metrics, decision_event, is_done = env.step(None)
        while not is_done:
            action = self._inference_agents.choose_action(decision_event, env.snapshot_list)
            metrics, decision_event, is_done = env.step(action)
            self._inference_agents.on_env_feedback(metrics)

        exp_by_agent = self._inference_agents.post_process(env.snapshot_list) if mode == RolloutMode.TRAIN else None
        performance = env.metrics
        env.reset()

        return {'local': performance}, exp_by_agent
