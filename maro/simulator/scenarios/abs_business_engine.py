# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from abc import ABC, abstractmethod
from typing import Dict, List

from maro.simulator.event_buffer import EventBuffer
from maro.simulator.frame import Frame, SnapshotList


class AbsBusinessEngine(ABC):
    """Abstract class for all the business engine

    Args:
        event_buffer (EventBuffer): used to process events
        topology_path (str): path to the topology folder
    """
    _event_buffer: EventBuffer

    def __init__(self, event_buffer: EventBuffer, config_path: str, start_tick: int, max_tick: int, frame_resolution: int):
        self._config_path = config_path
        self._event_buffer = event_buffer
        self._start_tick = start_tick
        self._max_tick = max_tick
        self._frame_resolution = frame_resolution

    @property
    @abstractmethod
    def frame(self) -> Frame:
        """Frame: Frame of current business engine
        """
        pass

    @property
    @abstractmethod
    def snapshots(self) -> SnapshotList:
        """SnapshotList: Snapshot list of current frame"""
        pass

    @abstractmethod
    def step(self, tick: int):
        """Used to process events at specified tick, usually this is called by Env at each tick

        Args:
            tick (int): tick to process
        """
        pass

    @property
    @abstractmethod
    def configs(self) -> dict:
        """object: Configurations of this business engine"""
        pass

    @abstractmethod
    def rewards(self, actions) -> float:
        """Calculate rewards based on actions

        Args:
            actions(list): Action(s) from agent

        Returns:
            float: reward based on actions
        """
        return []

    @abstractmethod
    def reset(self):
        """Reset business engine"""
        pass

    @abstractmethod
    def get_node_name_mapping(self) -> Dict[str, Dict]:
        """Get node name mappings related with this environment

        Returns:
            Dict[str, Dict]: Node name to index mapping dictionary
        """
        pass

    @abstractmethod
    def get_agent_idx_list(self) -> List[int]:
        """Get port index list related with this environment

        Returns:
            List[int]: list of port index
        """
        pass

    @abstractmethod
    def post_step(self, tick):
        """Post-process at specified tick

        Args:
            tick (int): tick to process
        """
        pass
