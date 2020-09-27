# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from abc import ABC, abstractmethod
from enum import IntEnum
from typing import List

from maro.backends.frame import SnapshotList
from maro.event_buffer import Event, EventBuffer
from maro.simulator.scenarios.abs_business_engine import AbsBusinessEngine


class DecisionMode(IntEnum):
    """Decision mode that interactive with agent."""

    # ask agent for action one by one
    Sequential = 0
    # ask agent for action at same time, not supported yes
    Joint = 1


class AbsEnv(ABC):
    """The main MARO simulator abstract class, which provides interfaces to agents.
    """

    def __init__(self, scenario: str, topology: str,
                 start_tick: int, durations: int, snapshot_resolution: int, max_snapshots: int,
                 decision_mode: DecisionMode,
                 business_engine_cls: type,
                 options: dict):
        """Create a new instance of environment

        Args:
            scenario (str): scenario name under maro/sim/scenarios folder
            topology (str): topology name under specified scenario folder
            start_tick (int): start tick of the scenario, usually used for pre-processed data streaming
            durations (int): duration ticks of this environment from start_tick
            snapshot_resolution (int): how many ticks will take a snapshot
            max_snapshots (int): max in-memory snapshot number, less snapshots lower memory cost
            business_engine_cls : class of business engine, if specified, then use it to construct be instance,
                or will search internal by scenario
            options (dict): additional parameters passed to business engine
        """
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

    @abstractmethod
    def step(self, action):
        """Push the environment to next step with action

        Args:
            action (Action): Action(s) from agent

        Returns:
            (float, object, bool): a tuple of (reward, decision event, is_done)

            The returned tuple contains 3 fields:

            - reward for current action. a list of reward if the input action is a list

            - decision_event for sequential decision mode, or a list of decision_event

            - whether the episode ends
        """
        pass

    @abstractmethod
    def dump(self):
        """Dump environment for restore"""
        pass

    @abstractmethod
    def reset(self):
        """Reset environment"""
        pass

    @property
    @abstractmethod
    def configs(self) -> dict:
        """object: Configurations of current environment,
        this field would be different for different scenario"""
        pass

    @property
    @abstractmethod
    def agent_idx_list(self) -> List[int]:
        """List[int]: Agent index list that related to this environment"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """str: Name of current environment"""
        pass

    @property
    @abstractmethod
    def tick(self) -> int:
        """int: Current tick of environment"""
        pass

    @property
    def frame_index(self) -> int:
        """int: frame index in snapshot list for current tick, USE this for snapshot querying"""
        pass

    @property
    @abstractmethod
    def summary(self) -> dict:
        """Summary about current simulator, may include node details, and mappings"""
        pass

    @property
    @abstractmethod
    def snapshot_list(self) -> SnapshotList:
        """Current snapshot list, a snapshot list contains all the snapshots of frame at each tick
        """
        pass

    def set_seed(self, seed: int):
        """Set random seed used by simulator.

        NOTE: this will not set seed for python random or other packages' seed, such as numpy.

        Args:
            seed (int):
        """

        pass

    @property
    def metrics(self) -> dict:
        """Some statistics information provided by business engine

        Returns:
            dict: dictionary of metrics, content and format is determined by business engine
        """

        return {}

    def get_finished_events(self) -> List[Event]:
        """List[Event]: All events finished so far
        """
        pass

    def get_pending_events(self, tick: int) -> List[Event]:
        """List[Event]: Pending events at certain tick

        Args:
            tick (int): Specified tick
        """
        pass
