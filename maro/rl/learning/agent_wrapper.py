# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict

from maro.rl.exploration import AbsExploration
from maro.rl.policy import AbsPolicy

from .env_wrapper import AbsEnvWrapper


class AgentWrapper:
    """Multi-agent wrapper that interacts with an ``EnvWrapper`` with a unified inferface.

    Args:
        policy_dict (Dict[str, AbsPolicy]): Policies for inference.
        agent2policy (Dict[str, str]): Mapping from agent ID's to policy ID's. This is used to direct an agent's
            queries to the correct policy.
        exploration_dict (Dict[str, AbsExploration]): A dictionary of named ``AbsExploration`` instances. Defaults
            to None.
        agent2exploration (Dict[str, str]): Mapping from agent names to exploration instance names. Defaults to None.
    """
    def __init__(
        self,
        policy_dict: Dict[str, AbsPolicy],
        agent2policy: Dict[str, str],
        exploration_dict: Dict[str, AbsExploration] = None,
        agent2exploration: Dict[str, str] = None
    ):
        self.policy_dict = policy_dict
        self.agent2policy = agent2policy
        self.policy = {agent_id: self.policy_dict[policy_id] for agent_id, policy_id in self.agent2policy.items()}
        self.exploration_dict = exploration_dict
        if self.exploration_dict:
            self.exploration_by_agent = {
                agent_id: exploration_dict[exploration_id] for agent_id, exploration_id in agent2exploration.items()
            }
        self.exploring = True  # Flag indicating that exploration is turned on.

    def choose_action(self, state: dict) -> dict:
        """Generate an action based on the given state.

        Args:
            state (dict): Dicitionary of agents' states based on which action decisions will be made.
        """
        action_by_agent = {agent_id: self.policy[agent_id].choose_action(st) for agent_id, st in state.items()}
        if self.exploring and self.exploration_dict:
            for agent_id in action_by_agent:
                if agent_id in self.exploration_by_agent:
                    action_by_agent[agent_id] = self.exploration_by_agent[agent_id](action_by_agent[agent_id])

        return action_by_agent

    def get_batch(self, env: AbsEnvWrapper):
        """Get experiences by policy names."""
        names = set()
        for agent_id, exp in env.get_experiences().items():
            if hasattr(self.policy[agent_id], "store"):
                self.policy[agent_id].store(exp)
            names.add(self.agent2policy[agent_id])

        return {name: self.policy_dict[name].experience_manager.get() for name in names}

    def set_policy_states(self, policy_state_dict: dict):
        """Update policy states."""
        for policy_id, policy_state in policy_state_dict.items():
            self.policy_dict[policy_id].set_state(policy_state)

    def exploration_step(self):
        if self.exploration_dict:
            for exploration in self.exploration_dict.values():
                exploration.step()

    def exploit(self):
        self.exploring = False

    def explore(self):
        self.exploring = True
