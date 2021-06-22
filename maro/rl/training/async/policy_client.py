# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
from os import getcwd
from typing import Dict

from maro.communication import Proxy, SessionMessage
from maro.rl.exploration import AbsExploration
from maro.utils import Logger

from ..message_enums import MsgKey, MsgTag


class PolicyClient:
    def __init__(
        self,
        agent2policy: Dict[str, str],
        group: str,
        exploration_dict: Dict[str, AbsExploration] = None,
        agent2exploration: Dict[str, str] = None,
        max_receive_attempts: int = None,
        receive_timeout: int = None,
        proxy_kwargs: dict = {},
        log_dir: str = getcwd()
    ):
        self.agent2policy = agent2policy
        self.exploration_dict = exploration_dict
        if self.exploration_dict:
            self.exploration_by_agent = {
                agent_id: exploration_dict[exploration_id] for agent_id, exploration_id in agent2exploration.items()
            }
        self.exploring = True  # Flag indicating that exploration is turned on.
        self._max_receive_attempts = max_receive_attempts
        self._receive_timeout = receive_timeout
        self._proxy = Proxy(group, "policy_client", {"inference_server": 1}, **proxy_kwargs)
        self._logger = Logger(self._proxy.name, dump_folder=log_dir)

    def choose_action(self, state_by_agent: dict, ep: int, step: int) -> dict:
        """Generate an action based on the given state.
        
        Args:
            state_by_agent (dict): Dicitionary of agents' states based on which action decisions will be made.
            ep (int): Current episode.
            step (int): Current step.
        """
        state_by_policy_name, agent_ids_by_policy_name = defaultdict(list), defaultdict(list)
        for agent_id, state in state_by_agent.items():
            policy_name = self.agent2policy[agent_id]
            state_by_policy_name[policy_name].append(state)
            agent_ids_by_policy_name[policy_name].append(agent_id)

        self._proxy.isend(
            SessionMessage(
                MsgTag.CHOOSE_ACTION, self._proxy.name, self._proxy.peers["policy_manager"][0],
                body={MsgKey.EPISODE: ep, MsgKey.STEP: step, MsgKey.STATE: dict(state_by_policy_name)}
            )
        )

        action_received = False
        for _ in range(self._max_receive_attempts):
            msg = self._proxy.receive_once(timeout=self._receive_timeout)
            if msg and msg.tag == MsgTag.ACTION and msg.body[MsgKey.EPISODE] == ep and msg.body[MsgKey.STEP] == step:
                action_received = True
                break

        if not action_received:
            self._logger(f"Failed to receive actions for episode {ep}, step {step}")
            return

        action_by_agent = {}
        for policy_name, action_batch in msg.body[MsgKey.ACTION].items():
            action_by_agent.update(dict(zip(agent_ids_by_policy_name[policy_name], action_batch)))

        if self.exploring and self.exploration_dict:
            for agent_id in action_by_agent:
                action_by_agent[agent_id] = self.exploration_by_agent[agent_id](action_by_agent[agent_id])

        return action_by_agent

    def exploration_step(self):
        for exploration in self.exploration_dict.values():
            exploration.step()
            print(f"epsilon: {exploration.epsilon}")

    def exploit(self):
        self.exploring = False

    def explore(self):
        self.exploring = True
