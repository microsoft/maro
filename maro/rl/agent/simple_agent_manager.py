# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod

from maro.rl.shaping.action_shaper import ActionShaper
from maro.rl.shaping.experience_shaper import ExperienceShaper
from maro.rl.shaping.state_shaper import StateShaper
from maro.rl.storage.column_based_store import ColumnBasedStore
from maro.utils.exception.rl_toolkit_exception import MissingShaperError

from .abs_agent_manager import AbsAgentManager, AgentManagerMode


class SimpleAgentManager(AbsAgentManager):
    def __init__(
        self,
        name: str,
        mode: AgentManagerMode,
        agent_dict: dict,
        state_shaper: StateShaper = None,
        action_shaper: ActionShaper = None,
        experience_shaper: ExperienceShaper = None
    ):
        if mode in {AgentManagerMode.INFERENCE, AgentManagerMode.TRAIN_INFERENCE}:
            if state_shaper is None:
                raise MissingShaperError(msg=f"state shaper cannot be None under mode {self._mode}")
            if action_shaper is None:
                raise MissingShaperError(msg=f"action_shaper cannot be None under mode {self._mode}")
            if experience_shaper is None:
                raise MissingShaperError(msg=f"experience_shaper cannot be None under mode {self._mode}")

        super().__init__(
            name, mode, agent_dict,
            state_shaper=state_shaper,
            action_shaper=action_shaper,
            experience_shaper=experience_shaper
        )
        # Data structures to temporarily store transitions and trajectory
        self._transition_cache = {}
        self._trajectory = ColumnBasedStore()

    def choose_action(self, decision_event, snapshot_list):
        self._assert_inference_mode()
        agent_id, model_state = self._state_shaper(decision_event, snapshot_list)
        model_action = self._agent_dict[agent_id].choose_action(model_state)
        self._transition_cache = {
            "state": model_state,
            "action": model_action,
            "reward": None,
            "agent_id": agent_id,
            "event": decision_event
        }
        return self._action_shaper(model_action, decision_event, snapshot_list)

    def on_env_feedback(self, metrics):
        """This method records the environment-generated metrics as part of the latest transition in the trajectory.

        Args:
            metrics: business metrics provided by the environment after an action has been executed.
        """
        self._transition_cache["metrics"] = metrics
        self._trajectory.put(self._transition_cache)

    def post_process(self, snapshot_list):
        """This method processes the latest trajectory into experiences.

        Args:
            snapshot_list: the snapshot list from the env at the end of an episode.
        """
        experiences = self._experience_shaper(self._trajectory, snapshot_list)
        self._trajectory.clear()
        self._transition_cache = {}
        self._state_shaper.reset()
        self._action_shaper.reset()
        self._experience_shaper.reset()
        return experiences

    @abstractmethod
    def train(self, experiences_by_agent: dict):
        """Train all agents."""
        return NotImplementedError
