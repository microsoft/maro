# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


class Transition:
    """Convenience class to be used in an environment wrapper's post-step processing function.

    Args:
        state: Output of the environment wrapper's ``get_state``.
        action: Output of the ``AgentWrapper`` that interacts with environment wrapper.
        env_action: Output of the environment wrapper's ``to_env_action``.
        reward: Output of the environmet wrapper's ``get_reward``.
        next_state: The state immediately following ``state``.
        info: Output of the environment wrapper's ``get_transition_info``.
    """

    __slots__ = ["state", "action", "env_action", "reward", "next_state", "info"]

    def __init__(self, state, action, env_action, reward, next_state, info):
        self.state = state
        self.action = action
        self.env_action = env_action
        self.reward = reward
        self.next_state = next_state
        self.info = info


class Trajectory:

    __slots__ = ["states", "actions", "rewards", "info", "length"]

    def __init__(self, states: list, actions: list, rewards: list, info: list):
        assert len(states) == len(actions) == len(rewards) + 1 == len(info) + 1
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.info = info
        self.length = len(rewards)
