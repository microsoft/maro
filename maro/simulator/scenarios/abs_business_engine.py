# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, List, Union

from maro.event_buffer import EventBuffer
from maro.backends.frame import FrameBase, SnapshotList
from maro.simulator.utils.common import tick_to_frame_index, total_frames


class AbsBusinessEngine(ABC):
    """Abstract class for all the business engine

    Args:
        event_buffer (EventBuffer): used to process events
        topology (str): config name
        start_tick (int): start tick of this business engine
        max_tick (int): max tick of this business engine
        snapshot_resolution (int): frequency to take a snapshot, NOTE: though we have this configuration, but business engine has the full control about the frequency of taking snapshot
        max_snapshots(int): max number of in-memory snapshots
        addition_options (dict): additional options for this business engine from outside
    """

    def __init__(self, scenario_name: str, event_buffer: EventBuffer, topology: str, start_tick: int, max_tick: int, snapshot_resolution: int, max_snapshots: int, additional_options: dict=None):
        self._scenario_name = scenario_name
        self._topology = topology
        self._event_buffer = event_buffer
        self._start_tick = start_tick
        self._max_tick = max_tick
        self._snapshot_resolution = snapshot_resolution
        self._max_snapshots = max_snapshots
        self._additional_options = additional_options
        self._config_path = None

        assert start_tick >= 0
        assert max_tick > start_tick
        assert max_snapshots is None or max_snapshots > 0

    @property
    @abstractmethod
    def frame(self) -> FrameBase:
        pass

    @property
    @abstractmethod
    def snapshots(self) -> SnapshotList:
        """SnapshotList: Snapshot list of current frame"""
        pass

    def frame_index(self, tick) -> int:
        """int: index of frame in snapshot list"""
        return tick_to_frame_index(self._start_tick, tick, self._snapshot_resolution)

    def calc_max_snapshots(self) -> int:
        """int: total snapshot should be in snapshot list.
        
        NOTE: this property will return max number that can contains all the frame state to the end.
        you can use a small size to hold states, when hit the limitation, oldest one will be overwrote
        """
        return  self._max_snapshots if self._max_snapshots is not None else total_frames(self._start_tick, self._max_tick, self._snapshot_resolution)

    def update_config_root_path(self, business_engine_file_path: str):
        """Update the config path with business engine path.

        Examples:

            .. code-block:: python

                # define a business engine
                class MyBusinessEngine(AbsBusinessEngine):
                    def __init__(self, *args, **kwargs):
                        super().__init__("my_be", *args, **kwargs)

                        # use __file__ as parameter
                        self.update_config_root_path(__file__)
        
        Args:
            business_engine_file_path(str): full path of real business engine file
        
        
        """
        if self._topology:
            path = Path(self._topology)
            
            if path.exists() and path.is_dir(): # and path.parent.name == "topologies":
                # if topology is a existing path, then use it as config root path
                self._config_path = self._topology
            else:
                self._config_path = os.path.join(os.path.split(os.path.realpath(business_engine_file_path))[0], "topologies", self._topology)

    @abstractmethod
    def step(self, tick: int):
        """Used to process events at specified tick, usually this is called by Env at each tick

        Args:
            tick (int): tick to process
        """
        pass

    @property
    def configs(self) -> dict:
        """object: Configurations of this business engine"""
        pass

    def rewards(self, actions) -> Union[float, List[float]]:
        """Calculate rewards based on actions

        Args:
            actions(list): Action(s) from agent

        Returns:
            Union[float, List[float]]: reward(s) based on actions
        """
        return []

    @abstractmethod
    def reset(self):
        """Reset business engine"""
        pass

    def post_step(self, tick:int) -> bool:
        """Post-process at specified tick, this function will be called at the end of each tick, 
        for complex business logic with many events, it maybe not easy to determine if stop the scenario at the middle of tick, so this method if used to avoid this.

        Args:
            tick (int): tick to process

        Returns:
            bool: if scenario end at this tick
        """
        return False

    def get_node_mapping(self) -> dict:
        """Get mapping for nodes, like index->name or index->id, may different for scenarios
        
        Returns:
            dict: key is node index, value is decided by scenario, usually is name or id
        """

        return {}

    def get_metrics(self) -> dict:
        """Get statistics information

        Returns:
            dict: dictionary about metrics, content and format determined by business engine
        
        """

        return {}