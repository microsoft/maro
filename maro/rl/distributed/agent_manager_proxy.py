# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict

from maro.communication import Proxy, SessionMessage
from maro.rl.agent_manager import AbsAgentManager
from maro.rl.shaping import Shaper

from .common import MessageTag, PayloadKey, TerminateEpisode


class AgentManagerProxy(AbsAgentManager):
    """
    A mirror of ``AgentManager`` that contains a proxy that obtains actions from a remote learner process and
    executes it locally.

    Args:
        state_shaper (Shaper, optional): It is responsible for converting the environment observation to model
            input.
        action_shaper (Shaper, optional): It is responsible for converting an agent's model output to environment
            executable action. Cannot be None under Inference and TrainInference modes.
        experience_shaper (Shaper, optional): It is responsible for processing data in the replay buffer at
            the end of an episode.
        max_receive_action_attempts (int): Maximum number of attempts to receive an action from the remote learner.
            Defaults to None, in which case, the proxy will keep trying to receive messages until the message containing
            the expected action is received.
    """
    def __init__(
        self,
        agent_proxy: Proxy,
        state_shaper: Shaper = None,
        action_shaper: Shaper = None,
        experience_shaper: Shaper = None,
        max_receive_action_attempts: int = None
    ):
        super().__init__(
            agent_proxy,
            state_shaper=state_shaper,
            action_shaper=action_shaper,
            experience_shaper=experience_shaper
        )
        self._max_receive_action_attempts = max_receive_action_attempts
        self._action_source = self.agents.peers_name["learner"][0]

        # Data structures to temporarily store trajectories
        self._trajectory = defaultdict(list)

        self._current_ep = None
        self._time_step = None

    @property
    def time_step(self):
        return self._time_step

    def reset(self, ep):
        self._current_ep = ep
        self._time_step = 0

    def choose_action(self, decision_event, snapshot_list):
        agent_id, model_state = self._state_shaper(decision_event, snapshot_list)
        action = self._query(*agent_id, model_state)
        if isinstance(action, TerminateEpisode):
            return action

        self._time_step += 1
        if action is None:
            return action

        self._transition_cache = {
            "state": model_state,
            "action": action,
            "reward": None,
            "agent_id": agent_id,
            "event": decision_event
        }
        return self._action_shaper(action, decision_event, snapshot_list)

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
        return experiences

    def _query(self, agent_id, model_state):
        payload = {
            PayloadKey.STATE: model_state,
            PayloadKey.AGENT_ID: agent_id,
            PayloadKey.EPISODE: self._current_ep,
            PayloadKey.TIME_STEP: self._time_step
        }
        self.agents.isend(
            SessionMessage(
                tag=MessageTag.CHOOSE_ACTION,
                source=self.agents.component_name,
                destination=self._action_source,
                payload=payload
            )
        )
        attempts_left = self._max_receive_action_attempts
        for msg in self.agents.receive():
            # Timeout
            if not msg:
                return
            if msg.tag == MessageTag.TERMINATE_EPISODE and msg.payload[PayloadKey.EPISODE] == self._current_ep:
                return TerminateEpisode()
            if msg.tag == MessageTag.ACTION:
                if (msg.payload[PayloadKey.EPISODE] == self._current_ep and
                        msg.payload[PayloadKey.TIME_STEP] == self._time_step):
                    return msg.payload[PayloadKey.ACTION]
            attempts_left -= 1
            if attempts_left == 0:
                return
