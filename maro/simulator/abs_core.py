# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from abc import ABC, abstractmethod
from enum import IntEnum
from typing import List

from maro.backends.frame import SnapshotList
from maro.event_buffer import EventBuffer
from maro.simulator.scenarios.abs_business_engine import AbsBusinessEngine


class DecisionMode(IntEnum):
    """Decision mode that interactive with agent."""
    # Ask agent to take action one by one.
    Sequential = 0
    # Ask agent to take action at same time, not supported yet.
    Joint = 1
    # Same as joint mode, by apply actions sequentially.
    JointWithSequentialAction = 2


class AbsEnv(ABC):
    """The main MARO simulator abstract class, which provides interfaces to agents.

        Args:
            scenario (str): Scenario name under maro/simulator/scenarios folder.
            topology (str): Topology name under specified scenario folder.
            start_tick (int): Start tick of the scenario, usually used for pre-processed data streaming.
            durations (int): Duration ticks of this environment from start_tick.
            snapshot_resolution (int): How many ticks will take a snapshot.
            max_snapshots (int): Max in-memory snapshot number, less snapshots lower memory cost.
            business_engine_cls(type): Class of business engine, if specified, then use it to construct be instance,
                or will search internal by scenario.
            disable_finished_events (bool): Disable finished events list, with this set to True, EventBuffer will
                re-use finished event object, this reduce event object number.
            options (dict): Additional parameters passed to business engine.
    """

    def __init__(
        self, scenario: str, topology: str,
        start_tick: int, durations: int, snapshot_resolution: int, max_snapshots: int,
        decision_mode: DecisionMode,
        business_engine_cls: type,
        disable_finished_events: bool,
        options: dict
    ):
        self._tick = start_tick
        self._scenario = scenario
        self._topology = topology
        self._start_tick = start_tick
        self._durations = durations
        self._snapshot_resolution = snapshot_resolution
        self._max_snapshots = max_snapshots
        self._decision_mode = decision_mode
        self._business_engine_cls = business_engine_cls
        self._additional_options = options

        self._business_engine: AbsBusinessEngine = None
        self._event_buffer: EventBuffer = None

    @property
    def business_engine(self):
        return self._business_engine

    @abstractmethod
    def step(self, action):
        """Push the environment to next step with action.

        Args:
            action (Action): Action(s) from agent.

        Returns:
            tuple: a tuple of (metrics, decision event, is_done).
        """
        pass

    @abstractmethod
    def dump(self):
        """Dump environment for restore."""
        pass

    @abstractmethod
    def reset(self):
        """Reset environment."""
        pass

    @property
    @abstractmethod
    def configs(self) -> dict:
        """object: Configurations of current environment, this field would be different for different scenario."""
        pass

    @property
    @abstractmethod
    def agent_idx_list(self) -> List[int]:
        """List[int]: Agent index list that related to this environment."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """str: Name of current environment."""
        pass

    @property
    @abstractmethod
    def tick(self) -> int:
        """int: Current tick of environment."""
        pass

    @property
    def frame_index(self) -> int:
        """int: Frame index in snapshot list for current tick, USE this for snapshot querying."""
        pass

    @property
    @abstractmethod
    def summary(self) -> dict:
        """dict: Summary about current simulator, may include node details, and mappings."""
        pass

    @property
    @abstractmethod
    def snapshot_list(self) -> SnapshotList:
        """SnapshotList: Current snapshot list, a snapshot list contains all the snapshots of frame at each tick."""
        pass

    def set_seed(self, seed: int):
        """Set random seed used by simulator.

        NOTE:
            This will not set seed for Python random or other packages' seed, such as NumPy.

        Args:
            seed (int): Seed to set.
        """
        pass

    @property
    def metrics(self) -> dict:
        """Some statistics information provided by business engine.

        Returns:
            dict: Dictionary of metrics, content and format is determined by business engine.
        """
        return {}

    def get_finished_events(self) -> list:
        """list: All events finished so far."""
        pass

    def get_pending_events(self, tick: int) -> list:
        """list: Pending events at certain tick.

        Args:
            tick (int): Specified tick.
        """
        pass
