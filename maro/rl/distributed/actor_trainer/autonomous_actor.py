# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict

from maro.communication import SessionMessage
from maro.rl.agent.simple_agent_manager import AbsAgentManager
from maro.rl.exploration.abs_explorer import AbsExplorer
from maro.rl.scheduling.scheduler import Scheduler
from maro.rl.shaping.state_shaper import StateShaper
from maro.rl.shaping.action_shaper import ActionShaper
from maro.rl.shaping.experience_shaper import ExperienceShaper
from maro.rl.storage.column_based_store import ColumnBasedStore
from maro.simulator import Env

from .abs_autonomous_actor import AbsAutoActor
from ..common import ActorTrainerComponent, MessageTag, PayloadKey


class SimpleAutoActor(AbsAutoActor):
    def __init__(self, env: Env, scheduler: Scheduler, agent_manager: AbsAgentManager, **proxy_params):
        super().__init__(env, scheduler, **proxy_params)
        self._agent_manager = agent_manager

    def run(self, is_training: bool = True):
        """Main loop for collecting experiences from the actor and using them to update policies."""
        for exploration_params in self._scheduler:
            self._env.reset()
            self._update_models()
            # load exploration parameters:
            if exploration_params is not None:
                self._agent_manager.update_exploration_params(exploration_params)

            metrics, decision_event, is_done = self._env.step(None)
            while not is_done:
                action = self._agent_manager.choose_action(decision_event, self._env.snapshot_list)
                metrics, decision_event, is_done = self._env.step(action)
                self._agent_manager.on_env_feedback(metrics)

            if is_training:
                self._scheduler.record_performance(self._env.metrics)
                experiences = self._agent_manager.post_process(self._env.snapshot_list)
                self._train(experiences)

    def _update_models(self):
        received = self._proxy.receive_by_id(
            [".".join([
                self._scheduler.current_ep,
                ActorTrainerComponent.ACTOR.value,
                ActorTrainerComponent.TRAINER.value
            ])]
        )
        self._agent_manager.load_models(received[0].payload[PayloadKey.MODEL])


class SEEDAutoActor(AbsAutoActor):
    def __init__(
        self,
        env: Env,
        scheduler: Scheduler,
        agent_id_list: list,
        state_shaper: StateShaper,
        action_shaper: ActionShaper,
        experience_shaper: ExperienceShaper,
        explorer: Dict[str, AbsExplorer],
        **proxy_params
    ):
        super().__init__(env, scheduler, **proxy_params)
        self._agent_id_set = set(agent_id_list)
        self._state_shaper = state_shaper
        self._action_shaper = action_shaper
        self._experience_shaper = experience_shaper
        self._explorer = explorer
        # Data structures to temporarily store transitions and trajectory
        self._transition_cache = {}
        self._trajectory = ColumnBasedStore()

    def run(self, is_training: bool = True):
        """Main loop for collecting experiences from the actor and using them to update policies."""
        for exploration_params in self._scheduler:
            self._env.reset()
            self._update_exploration_params(exploration_params)

            metrics, decision_event, is_done = self._env.step(None)
            while not is_done:
                action = self._choose_action(decision_event, self._env.snapshot_list)
                metrics, decision_event, is_done = self._env.step(action)
                if is_training:
                    self._transition_cache["metrics"] = metrics
                    self._trajectory.put(self._transition_cache)

            if is_training:
                experiences = self._post_process()
                self._scheduler.record_performance(self._env.metrics)
                self._update(experiences)

    def _update_exploration_params(self, exploration_params):
        # Per-agent exploration parameters
        if isinstance(exploration_params, dict) and exploration_params.keys() <= self._agent_id_set:
            for agent_id, params in exploration_params.items():
                self._explorer[agent_id].update(**params)
        # Shared exploration parameters for all agents
        else:
            for explorer in self._explorer.values():
                explorer.update(**exploration_params)

    def _choose_action(self, decision_event, snapshot_list):
        agent_id, model_state = self._state_shaper(decision_event, snapshot_list)
        reply = self._proxy.send(
            SessionMessage(
                tag=MessageTag.CHOOSE_ACTION,
                source=self._proxy.component_name,
                destination=self._proxy.peers_name[ActorTrainerComponent.TRAINER.value][0],
                payload={PayloadKey.STATE: model_state, PayloadKey.AGENT_ID: agent_id},
            )
        )
        model_action = self._explorer[agent_id](reply[0].payload[PayloadKey.ACTION])
        self._transition_cache = {
            "state": model_state,
            "action": model_action,
            "reward": None,
            "agent_id": agent_id,
            "event": decision_event
        }
        return self._action_shaper(model_action, decision_event, snapshot_list)

    def _post_process(self):
        """Process the latest trajectory into experiences."""
        experiences = self._experience_shaper(self._trajectory, self._env.snapshot_list)
        self._trajectory.clear()
        self._transition_cache = {}
        self._state_shaper.reset()
        self._action_shaper.reset()
        self._experience_shaper.reset()
        return experiences
