# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
from typing import Dict

from maro.rl.policy import AbsPolicy, RLPolicy

from .env_wrapper import Trajectory


class AgentWrapper:
    """Multi-agent wrapper that interacts with an ``EnvWrapper`` with a unified inferface.

    Args:
        create_policy_func_dict (dict): A dictionary mapping policy names to functions that create them. The policy
            creation function should have policy name as the only parameter and return an ``AbsPolicy`` instance.
        agent2policy (Dict[str, str]): Mapping from agent ID's to policy ID's. This is used to direct an agent's
            queries to the correct policy.
    """
    def __init__(
        self,
        create_policy_func_dict: Dict[str, AbsPolicy],
        agent2policy: Dict[str, str]
    ):
        self.policy_dict = {name: func(name) for name, func in create_policy_func_dict.items()}
        self.agent2policy = agent2policy
        self.policy = {
            agent_id: self.policy_dict[policy_id] for agent_id, policy_id in self.agent2policy.items()
        }

    def choose_action(self, state: dict) -> dict:
        """Generate an action based on the given state.

        Args:
            state (dict): Dictionary of agents' states based on which action decisions will be made.
        """
        return {agent_id: self.policy[agent_id].choose_action(st) for agent_id, st in state.items()}

    def get_rollout_info(self, trajectory: Dict[str, Trajectory]):
        """Get experiences by policy names."""
        rollout_info = {}
        for agent_id, traj in trajectory.items():
            if isinstance(self.policy[agent_id], RLPolicy):
                policy_name = self.agent2policy[agent_id]
                rollout_info[policy_name] = self.policy_dict[policy_name].get_rollout_info(traj)

        return rollout_info

    def set_policy_states(self, policy_state_dict: dict):
        """Update policy states."""
        for policy_id, policy_state in policy_state_dict.items():
            self.policy_dict[policy_id].set_state(policy_state)

    def exploration_step(self):
        for policy in self.policy_dict.values():
            if hasattr(policy, "exploration_step"):
                policy.exploration_step()

    def exploit(self):
        for policy in self.policy_dict.values():
            if hasattr(policy, "exploit"):
                policy.exploit()

    def explore(self):
        for policy in self.policy_dict.values():
            if hasattr(policy, "explore"):
                policy.explore()
