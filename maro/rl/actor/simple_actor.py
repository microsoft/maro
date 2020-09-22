# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .abs_actor import AbsActor
from maro.rl.agent.abs_agent_manager import AbsAgentManager
from maro.simulator import Env


class SimpleActor(AbsActor):
    """
    A simple actor class that implements simple roll-out logic
    """
    def __init__(self, env: Env, inference_agents: AbsAgentManager):
        super().__init__(env, inference_agents)

    def roll_out(self, model_dict: dict = None, epsilon_dict: dict = None, done: bool = False,
                 return_details: bool = True):
        """
        The main interface provided by the Actor class, in which the agents perform a single episode of roll-out
        to collect experiences and performance data from the environment

        Args:
            model_dict (dict): if not None, the agents will load the models from model_dict and use these models
                           to perform roll-out.
            epsilon_dict (dict): exploration rate by agent
            done (bool): if True, the current call is the last call, i.e., no more roll-outs will be performed.
                         This flag is used to signal remote actor workers to exit.
            return_details (bool): if True, return experiences as well as performance metrics provided by the env.
        """
        if done:
            return None, None

        self._env.reset()
        # assign epsilons
        if epsilon_dict is not None:
            self._inference_agents.explorer.epsilon = epsilon_dict

        # load models
        if model_dict is not None:
            self._inference_agents.load_models(model_dict)

        metrics, decision_event, is_done = self._env.step(None)
        while not is_done:
            action = self._inference_agents.choose_action(decision_event, self._env.snapshot_list)
            metrics, decision_event, is_done = self._env.step(action)
            self._inference_agents.on_env_feedback(metrics)

        details = self._inference_agents.post_process(self._env.snapshot_list) if return_details else None

        return self._env.metrics, details
