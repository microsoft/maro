# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.rl.agent_manager import AbsAgentManager
from maro.simulator import Env

from maro.rl.actor import AbsActor
from maro.simulator.scenarios.vm_scheduling import AllocateAction, DecisionPayload, PostponeAction


class VMActor(AbsActor):
    """A simple ``AbsActor`` implementation.

    Args:
        env (Env): An Env instance.
        agent_manager (AbsAgentManager): An AgentManager instance that manages all agents.
    """
    def __init__(self, env: Env, agent_manager: AbsAgentManager):
        super().__init__(env, agent_manager)

    def roll_out(
        self, model_dict: dict = None, exploration_params=None, done: bool = False, return_details: bool = True
    ):
        """Perform one episode of roll-out and return performance and experiences.

        Args:
            model_dict (dict): If not None, the agents will load the models from model_dict and use these models
                to perform roll-out.
            exploration_params: Exploration parameters.
            done (bool): If True, the current call is the last call, i.e., no more roll-outs will be performed.
                This flag is used to signal remote actor workers to exit.
            return_details (bool): If True, return experiences as well as performance metrics provided by the env.

        Returns:
            Performance and relevant details from the episode (e.g., experiences).
        """
        if done:
            return None, None

        self._env.reset()

        # load models
        if model_dict is not None:
            self.agent_manager.load_models(model_dict)

        # load exploration parameters:
        if exploration_params is not None:
            self.agent_manager.set_exploration_params(exploration_params)
        else:
            self.agent_manager.set_exploration_params({'epsilon': 0.0})

        reward = 0.0
        metrics, decision_event, is_done = self._env.step(None)
        vm_num = 0
        while not is_done:
            vm_num += 1
            action, postpone = self.agent_manager.choose_action(decision_event, self._env)
            if postpone:
                reward -= 0.1
            else:
                reward += 1.0
            metrics, decision_event, is_done = self._env.step(action)
            self.agent_manager.on_env_feedback(metrics)

        details = self.agent_manager.post_process(return_details, reward / vm_num)

        return self._env.metrics, details, reward
