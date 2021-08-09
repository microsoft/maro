# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict

from maro.rl.policy import AbsPolicy

from .env_wrapper import AbsEnvWrapper


class AgentWrapper:
    """Multi-agent wrapper that interacts with an ``EnvWrapper`` with a unified inferface.

    Args:
        policy_dict (Dict[str, AbsPolicy]): Policies used by the agents to make decision when interacting with
            the environment.
        agent2policy (Dict[str, str]): Mapping from agent ID's to policy ID's. This is used to direct an agent's
            queries to the correct policy.
    """
    def __init__(self, policy_dict: Dict[str, AbsPolicy], agent2policy: Dict[str, str]):
        self.policy_dict = policy_dict
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

    def get_batch(self, env: AbsEnvWrapper):
        """Get experiences by policy names."""
        names = set()
        exp_by_agent = env.get_experiences()
        for agent_id, exp in exp_by_agent.items():
            if hasattr(self.policy[agent_id], "memorize"):
                self.policy[agent_id].memorize(exp)
            names.add(self.agent2policy[agent_id])

        ret = {name: self.policy_dict[name].sampler.get() for name in names}
        print({name: batch.data.size for name, batch in ret.items()})
        return ret

    def set_policy_states(self, policy_state_dict: dict):
        """Update policy states."""
        for policy_id, policy_state in policy_state_dict.items():
            self.policy_dict[policy_id].algorithm.set_state(policy_state)

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
