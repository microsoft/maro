from copy import deepcopy
from typing import List, Optional

import gym
from maro.backends.frame import FrameBase, SnapshotList

from maro.event_buffer import CascadeEvent, EventBuffer, MaroEvents
from maro.simulator.scenarios import AbsBusinessEngine


class GymBusinessEngine(AbsBusinessEngine):
    def __init__(
        self, event_buffer: EventBuffer, topology: Optional[str], start_tick: int,
        max_tick: int, snapshot_resolution: int, max_snapshots: Optional[int], additional_options: dict = None,
    ) -> None:
        super(GymBusinessEngine, self).__init__(
            scenario_name="gym", event_buffer=event_buffer, topology=topology, start_tick=start_tick,
            max_tick=max_tick, snapshot_resolution=snapshot_resolution, max_snapshots=max_snapshots,
            additional_options=additional_options,
        )

        self._gym_scenario_name = "Walker2d-v4"
        self._gym_env = gym.make(self._gym_scenario_name)
        self._seed = additional_options.get("random_seed", None)

        self._last_obs = self._gym_env.reset()[0]
        self._is_done = False
        self._reward_record = {}
        self._info_record = {}

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
        actions = event.payload
        assert isinstance(actions, list)
        action = actions[0]

        self._last_obs, reward, self._is_done, _, info = self._gym_env.step(action)
        self._reward_record[event.tick] = reward
        self._info_record[event.tick] = info

    def step(self, tick: int) -> None:
        self._event_buffer.insert_event(self._event_buffer.gen_decision_event(tick, self._last_obs))

    @property
    def configs(self) -> dict:
        return {}

    def get_reward_at_tick(self, tick: int) -> float:
        return self._reward_record[tick]

    def get_info_at_tick(self, tick: int) -> object:  # TODO
        return self._info_record[tick]

    def reset(self, keep_seed: bool = False) -> None:
        self._last_obs = self._gym_env.reset()[0]
        self._is_done = False
        self._reward_record = {}
        self._info_record = {}

    def post_step(self, tick: int) -> bool:
        return self._is_done or tick + 1 == self._max_tick

    def get_agent_idx_list(self) -> List[int]:
        return [0]

    def get_metrics(self) -> dict:
        return {
            "reward_record": {k: v for k, v in self._reward_record.items()},
        }
