# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Optional, cast

import gym
import numpy as np

from maro.backends.frame import FrameBase, SnapshotList
from maro.event_buffer import CascadeEvent, EventBuffer, MaroEvents
from maro.simulator.scenarios import AbsBusinessEngine

from .common import Action, DecisionEvent


class GymBusinessEngine(AbsBusinessEngine):
    def __init__(
        self,
        event_buffer: EventBuffer,
        topology: Optional[str],
        start_tick: int,
        max_tick: int,
        snapshot_resolution: int,
        max_snapshots: Optional[int],
        additional_options: dict = None,
    ) -> None:
        super(GymBusinessEngine, self).__init__(
            scenario_name="gym",
            event_buffer=event_buffer,
            topology=topology,
            start_tick=start_tick,
            max_tick=max_tick,
            snapshot_resolution=snapshot_resolution,
            max_snapshots=max_snapshots,
            additional_options=additional_options,
        )

        self._gym_scenario_name = topology
        self._gym_env = gym.make(self._gym_scenario_name)

        self.reset()

        self._frame: FrameBase = FrameBase()
        self._snapshots: SnapshotList = self._frame.snapshots

        self._register_events()

    @property
    def gym_env(self) -> gym.Env:
        return self._gym_env

    @property
    def frame(self) -> FrameBase:
        return self._frame

    @property
    def snapshots(self) -> SnapshotList:
        return self._snapshots

    def _register_events(self) -> None:
        self._event_buffer.register_event_handler(MaroEvents.TAKE_ACTION, self._on_action_received)

    def _on_action_received(self, event: CascadeEvent) -> None:
        action = cast(Action, cast(list, event.payload)[0]).action

        self._last_obs, reward, self._is_done, self._truncated, info = self._gym_env.step(action)
        self._reward_record[event.tick] = reward
        self._info_record[event.tick] = info

    def step(self, tick: int) -> None:
        self._event_buffer.insert_event(self._event_buffer.gen_decision_event(tick, DecisionEvent(self._last_obs)))

    @property
    def configs(self) -> dict:
        return {}

    def get_reward_at_tick(self, tick: int) -> float:
        return self._reward_record[tick]

    def get_info_at_tick(self, tick: int) -> object:  # TODO
        return self._info_record[tick]

    def reset(self, keep_seed: bool = False) -> None:
        self._last_obs = self._gym_env.reset(seed=np.random.randint(low=0, high=4096))[0]
        self._is_done = False
        self._truncated = False
        self._reward_record = {}
        self._info_record = {}

    def post_step(self, tick: int) -> bool:
        return self._is_done or self._truncated or tick + 1 == self._max_tick

    def get_agent_idx_list(self) -> List[int]:
        return [0]

    def get_metrics(self) -> dict:
        return {
            "reward_record": {k: v for k, v in self._reward_record.items()},
        }

    def set_seed(self, seed: int) -> None:
        pass
