# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable

from maro.rl.wrappers import AbsEnvWrapper, AgentWrapper


class RolloutWorker:
    def __init__(
        self,
        get_env_wrapper: Callable[[], AbsEnvWrapper],
        get_agent_wrapper: Callable[[], AgentWrapper],
        get_eval_env_wrapper: Callable[[], AbsEnvWrapper] = None
    ):
        self.env = get_env_wrapper()
        self.eval_env = get_env_wrapper() if get_eval_env_wrapper else self.env
        self.agent = get_agent_wrapper()

    def sample(self, policy_state_dict: dict = None, num_steps: int = -1, exploration_step: bool = False):
        # set policy states
        if policy_state_dict:
            self.agent.set_policy_states(policy_state_dict)

        # set exploration parameters
        self.agent.explore()
        if exploration_step:
            self.agent.exploration_step()

        if self.env.state is None:
            self.env.reset()
            self.env.replay = True
            self.env.start()  # get initial state

        starting_step_index = self.env.step_index + 1
        steps_to_go = float("inf") if num_steps == -1 else num_steps
        while self.env.state and steps_to_go > 0:
            action = self.agent.choose_action(self.env.state)
            self.env.step(action)
            steps_to_go -= 1

        return {
            "rollout_info": self.agent.get_rollout_info(self.env.get_trajectory()),
            "step_range": (starting_step_index, self.env.step_index),
            "tracker": self.env.tracker,
            "end_of_episode": not self.env.state,
            "exploration_params": self.agent.exploration_params
        }

    def test(self, policy_state_dict: dict):
        self.agent.set_policy_states(policy_state_dict)
        self.agent.exploit()
        self.eval_env.reset()
        self.eval_env.replay = False
        self.eval_env.start()  # get initial state
        while self.eval_env.state:
            action = self.agent.choose_action(self.eval_env.state)
            self.eval_env.step(action)

        return self.eval_env.tracker
