# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod

from maro.communication import Proxy, SessionMessage
from maro.rl.shaping.action_shaper import ActionShaper
from maro.rl.shaping.experience_shaper import ExperienceShaper
from maro.rl.shaping.state_shaper import StateShaper
from maro.rl.storage.column_based_store import ColumnBasedStore

from .common import MessageTag, PayloadKey


class Executor(object):
    """An ``Executor`` is responsible for interacting with an environment.

    An ``Executor`` consists of a state shaper for observing the environment and an action shaper for executing
    actions on it. It also has an experience shaper that processes trajectories into experiences for remote training.

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
        state_shaper: StateShaper,
        action_shaper: ActionShaper,
        experience_shaper: ExperienceShaper
    ):
        self._state_shaper = state_shaper
        self._action_shaper = action_shaper
        self._experience_shaper = experience_shaper

        # Data structures to temporarily store transitions and trajectory
        self._transition_cache = {}
        self._trajectory = ColumnBasedStore()

        self._proxy = None

    def load_proxy(self, proxy: Proxy):
        self._proxy = proxy

    def choose_action(self, decision_event, snapshot_list):
        assert self._proxy is not None, "A proxy needs to be loaded first by calling load_proxy()"
        remote_action_source = self._proxy.peers_name[list(self._proxy.peers_name.keys())[0]][0]
        agent_id, model_state = self._state_shaper(decision_event, snapshot_list)
        reply = self._proxy.send(
            SessionMessage(
                tag=MessageTag.CHOOSE_ACTION,
                source=self._proxy.component_name,
                destination=remote_action_source,
                payload={PayloadKey.STATE: model_state, PayloadKey.AGENT_ID: agent_id},
            )
        )
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
        self._transition_cache = {}
        self._state_shaper.reset()
        self._action_shaper.reset()
        self._experience_shaper.reset()
        return experiences
