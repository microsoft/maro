# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from typing import List, Optional, cast
from yaml import safe_load

from maro.backends.frame import FrameBase, SnapshotList
from maro.event_buffer import CascadeEvent, EventBuffer, MaroEvents
from maro.simulator.scenarios import AbsBusinessEngine

from .common import Action, DecisionEvent


class SimpleRacingBusinessEngine(AbsBusinessEngine):
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
        super(SimpleRacingBusinessEngine, self).__init__(
            scenario_name="simple_racing",
            event_buffer=event_buffer,
            topology=topology,
            start_tick=start_tick,
            max_tick=max_tick,
            snapshot_resolution=snapshot_resolution,
            max_snapshots=max_snapshots,
            additional_options=additional_options,
        )
        self._config: dict = self._load_configs()

        # TODO: Any other required parameters given by the function call can be get like the random_seed below.
        self._seed = additional_options.get("random_seed", None)

        # TODO: Add your initialization logic here. Or just call your current initialization function here.

        self.reset()

        self._frame: FrameBase = FrameBase()
        self._snapshots: SnapshotList = self._frame.snapshots

        self._register_events()

    def _load_configs(self) -> dict:
        """Load configurations"""
        self.update_config_root_path(__file__)

        config = {}
        with open(os.path.join(self._config_path, "config.yml")) as fp:
            config = safe_load(fp)

        # TODO: Add the processing logic for the information in yaml config here, if any.

        return config

    @property
    def frame(self) -> FrameBase:
        return self._frame

    @property
    def snapshots(self) -> SnapshotList:
        # TODO: We do suggest you to wrap your data that is accessed frequently with the NodeBase and NodeAttribute
        # wrapper, as we show in the jupyter notebook, to actually utilize the underlying efficient data model of MARO.
        return self._snapshots

    def _register_events(self) -> None:
        self._event_buffer.register_event_handler(MaroEvents.TAKE_ACTION, self._on_action_received)

    def _on_action_received(self, event: CascadeEvent) -> None:
        # TODO: We do suggest you to define your Action as a class to maintain richer information.
        # It is also a decomposition of simulator implementation and RL problem formulation.
        # Here the variable action is a np.ndarray as other Gym Env.
        action = cast(Action, cast(list, event.payload)[0]).action

        # TODO: add the logic of the step() function for Gym Env here. Or just call your step() function here.
        # If the immediate reward can be defined like other Gym Env, you can call your gym-like env as:
        self._last_obs, reward, self._is_done, self._truncated, info = YourGymEnv.step(action)
        self._reward_record[event.tick] = reward
        self._info_record[event.tick] = info

    def step(self, tick: int) -> None:
        # TODO: We do suggest you to define your DecisionEvent as a class to maintain richer information.
        # It is also a decomposition of simulator implementation and RL problem formulation.
        # Here the parameter of the DecisionEvent instance (self._last_obs) is a np.ndarray (state) as other Gym Env.
        self._event_buffer.insert_event(self._event_buffer.gen_decision_event(tick, DecisionEvent(self._last_obs)))

    @property
    def configs(self) -> dict:
        return {}

    def get_reward_at_tick(self, tick: int) -> float:
        # TODO: This function is not necessary if you implement a complex reward-shaping function in EnvSampler.
        # Keep it if you define immediate reward in function self._on_action_received(event)
        return self._reward_record[tick]

    def get_info_at_tick(self, tick: int) -> object:
        # TODO: This function is not necessary if you implement complex shaping functions in EnvSampler.
        # Keep it if you define the info in function self._on_action_received(event)
        return self._info_record[tick]

    def reset(self, keep_seed: bool = False) -> None:
        # TODO: Add your reset logic here. Or just call your current reset function here.
        self._last_obs = YourGymEnv.reset()
        self._is_done = False
        self._truncated = False
        self._reward_record = {}
        self._info_record = {}

    def post_step(self, tick: int) -> bool:
        return self._is_done or self._truncated or tick + 1 == self._max_tick

    def get_agent_idx_list(self) -> List[int]:
        # TODO: Add your agent list here. The simplest case is to map each car to an identical id/idx.
        return [0]

    def get_metrics(self) -> dict:
        # TODO: Add other metrics here if have any.
        return {
            "reward_record": {k: v for k, v in self._reward_record.items()},
        }

    def set_seed(self, seed: int) -> None:
        pass
