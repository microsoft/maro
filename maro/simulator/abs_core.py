# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from abc import ABC, abstractmethod
from .graph import SnapshotList, Graph
from enum import IntEnum
from typing import Dict, Tuple, List, Any
from maro.simulator.event_buffer import Event, EventBuffer
from maro.simulator.scenarios.abs_business_engine import AbsBusinessEngine


class DecisionMode(IntEnum):
    """Decision mode that interactive with agent."""
    Sequential = 0  # ask agent for action one by one
    Joint = 1  # ask agent for action at same time, not supported yes


class AbsEnv(ABC):
    """The main MARO simulator abstract class, which provides interfaces to agents.
    """

    def __init__(self, scenario: str, topology: str, max_tick: int = 100, tick_unit: int = 1, decision_mode=DecisionMode.Sequential):
        """Create a new instance of environment

        Args:
            scenario (str): scenario name under maro/sim/scenarios folder
            topology (topology): topology name under specified scenario folder
            max_tick (int): max tick of this environment
            tick_unit (int): used to control tick unit, the size of snapshot will be same with max_tick, other part with use the tick unit to push the simulator
        """
        self._tick = 0
        self._scenario = scenario
        self._topology = topology
        self._max_tick = max_tick
        self._tick_units = tick_unit
        self._decision_mode = decision_mode
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

            - decision_event for sequential decision mode, or a list of decision_event (the pending event can be any object,
              like DecisionEvent for ECR scenario)"

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
        """object: Configurations of current environment, this field would be different for different scenario"""
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
    @abstractmethod
    def node_name_mapping(self) -> Dict[str, List]:
        """Dict[str, List]: Resource node name mapping that configured for current environment"""
        pass

    @property
    @abstractmethod
    def snapshot_list(self) -> SnapshotList:
        """Current snapshot list, a snapshot list contains all the snapshots of graph at each tick
        """
        pass

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
