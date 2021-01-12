# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Union

from maro.communication import Message, Proxy, SessionMessage
from maro.rl.agent import AbsAgentManager
from maro.rl.shaping.action_shaper import ActionShaper
from maro.rl.shaping.experience_shaper import ExperienceShaper
from maro.rl.shaping.state_shaper import StateShaper
from maro.rl.storage.column_based_store import ColumnBasedStore

from .common import MessageTag, PayloadKey, TerminateEpisode


class AgentManagerProxy(AbsAgentManager):
    """
    A mirror of ``AgentManager`` that contains a proxy that obtains actions from a remote learner process and
    executes it locally.

    Args:
        state_shaper (StateShaper, optional): It is responsible for converting the environment observation to model
            input.
        action_shaper (ActionShaper, optional): It is responsible for converting an agent's model output to environment
            executable action. Cannot be None under Inference and TrainInference modes.
        experience_shaper (ExperienceShaper, optional): It is responsible for processing data in the replay buffer at
            the end of an episode.
    """
    def __init__(
        self,
        agent_proxy: Proxy,
        state_shaper: StateShaper = None,
        action_shaper: ActionShaper = None,
        experience_shaper: ExperienceShaper = None,
        action_wait_timeout: int = None
    ):
        super().__init__(
            agent_proxy, 
            state_shaper=state_shaper, 
            action_shaper=action_shaper,
            experience_shaper=experience_shaper
        )
        self._action_wait_timeout = action_wait_timeout
        self._action_source = self.agents.peers_name["learner"][0]

        # Data structures to temporarily store transitions and trajectory
        self._transition_cache = {}
        self._trajectory = ColumnBasedStore()

        self._current_ep = None

    def set_ep(self, ep):
        self._current_ep = ep

    def choose_action(self, decision_event, snapshot_list):
        # Check if a TerminateEpisode signal was received
        for msg in self.agents.get_by_tag(MessageTag.TERMINATE_EPISODE):
            if msg.payload[PayloadKey.EPISODE] == self._current_ep:
                return TerminateEpisode()
        agent_id, model_state = self._state_shaper(decision_event, snapshot_list)
        payload = {
            PayloadKey.STATE: model_state, 
            PayloadKey.AGENT_ID: agent_id,
            PayloadKey.EPISODE: self._current_ep,
        }
        reply = self.agents.send(
            SessionMessage(
                tag=MessageTag.CHOOSE_ACTION,
                source=self.agents.component_name,
                destination=self._action_source,
                payload=payload
            ),
            timeout=self._action_wait_timeout
        )
        # Timeout
        if not reply:
            return

        model_action = reply[0].payload[PayloadKey.ACTION]
        self._transition_cache = {
            "state": model_state,
            "action": model_action,
            "reward": None,
            "agent_id": agent_id,
            "event": decision_event
        }
        return self._action_shaper(model_action, decision_event, snapshot_list)

    def on_env_feedback(self, metrics):
        self._transition_cache["metrics"] = metrics
        self._trajectory.put(self._transition_cache)

    def post_process(self, snapshot_list):
        """Process the latest trajectory into experiences."""
        experiences = self._experience_shaper(self._trajectory, snapshot_list)
        self._trajectory.clear()
        self._transition_cache.clear()
        self._state_shaper.reset()
        self._action_shaper.reset()
        self._experience_shaper.reset()
        self.agents.purge()
        return experiences
