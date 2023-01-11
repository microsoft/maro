from typing import Dict, List, Optional

import gym

from maro.backends.frame import FrameBase, SnapshotList
from maro.event_buffer import CascadeEvent, EventBuffer, MaroEvents
from maro.simulator.scenarios import AbsBusinessEngine

from .common import Action, DecisionEvent


class GymBusinessEngine(AbsBusinessEngine):
    def __init__(
        self, event_buffer: EventBuffer, topology: str, start_tick: int, max_tick: int,
        snapshot_resolution: int, max_snapshots: Optional[int], additional_options: dict = None,
    ) -> None:
        super(GymBusinessEngine, self).__init__(
            scenario_name="gym", event_buffer=event_buffer, topology=topology, start_tick=start_tick,
            max_tick=max_tick, snapshot_resolution=snapshot_resolution, max_snapshots=max_snapshots,
            additional_options=additional_options,
        )

        self._gym_scenario_name = topology
        self._gym_env = gym.make(self._gym_scenario_name)

        self._seed = additional_options.get("random_seed", None)

        self._frame: FrameBase = FrameBase()
        self._snapshots: SnapshotList = self._frame.snapshots
        self._register_events()

        self._last_obs: object = None
        self._terminated: bool = False
        self._truncated: bool = False
        self._reward_record: Dict[int, float] = {}
        self._info_record: Dict[int, dict] = {}
        self.reset()

    @property
    def gym_env(self) -> gym.Env:
        return self._gym_env

    @property
    def frame(self) -> FrameBase:
        """Abstract Method of AbsBusinessEngine."""
        return self._frame

    @property
    def snapshots(self) -> SnapshotList:
        """Abstract Method of AbsBusinessEngine."""
        return self._snapshots

    @property
    def configs(self) -> dict:
        """Abstract Method of AbsBusinessEngine. Not used here."""
        return {}

    def get_agent_idx_list(self) -> List[int]:
        """Abstract Method of AbsBusinessEngine. 1 agent Only here."""
        return [0]

    def set_seed(self, seed: int) -> None:
        """Abstract Method of AbsBusinessEngine. Not used here."""
        raise NotImplementedError("Gym Env can only set seed in reset()")

    def _register_events(self) -> None:
        self._event_buffer.register_event_handler(MaroEvents.TAKE_ACTION, self._on_action_received)

    def _on_action_received(self, event: CascadeEvent) -> None:
        actions = event.payload
        assert isinstance(actions, list)
        assert isinstance(actions[0], Action)
        action = actions[0].action

        self._last_obs, reward, self._terminated, self._truncated, info = self._gym_env.step(action)
        self._reward_record[event.tick] = reward
        self._info_record[event.tick] = info

    def step(self, tick: int) -> None:
        self._event_buffer.insert_event(self._event_buffer.gen_decision_event(tick, DecisionEvent(self._last_obs)))

    def post_step(self, tick: int) -> bool:
        return self._terminated or self._truncated or tick + 1 == self._max_tick

    def reset(self, keep_seed: bool = False) -> None:
        self._last_obs, _ = self._gym_env.reset(seed=self._seed)
        self._terminated = False
        self._reward_record = {}
        self._info_record = {}

    def get_metrics(self) -> dict:
        return {
            "reward_record": {k: v for k, v in self._reward_record.items()},
        }

    def get_reward_at_tick(self, tick: int) -> float:
        return self._reward_record[tick]

    def get_info_at_tick(self, tick: int) -> object:  # TODO
        return self._info_record[tick]
