# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from abc import ABC, abstractmethod
from pathlib import Path

from maro.backends.frame import FrameBase, SnapshotList
from maro.event_buffer import EventBuffer
from maro.simulator.utils.common import tick_to_frame_index, total_frames


class AbsBusinessEngine(ABC):
    """Abstract class for all the business engine, a business engine is the core part of a scenario,
    used to hold all related logics.

    A business engine should have a name that used to identify it, built-in scenarios also use it to
    find built-in topologies.

    The core part of business engine is the step and post_step methods:

    1. step: Will be called one time at each tick.
    2. post_step: Will be called at the end of each tick after all the events being processed, \
    simulator use the return value of this method (bool), to decide if it should stop simulation. \
    This is also a good place to check business final state of current tick if you follow event-driven pattern.

    Args:
        event_buffer (EventBuffer): Used to process events.
        topology (str): Config name.
        start_tick (int): Start tick of this business engine.
        max_tick (int): Max tick of this business engine.
        snapshot_resolution (int): Frequency to take a snapshot.
        max_snapshots(int): Max number of in-memory snapshots, default is None that means max number of snapshots.
        addition_options (dict): Additional options for this business engine from outside.
    """

    def __init__(
        self, scenario_name: str, event_buffer: EventBuffer, topology: str,
        start_tick: int, max_tick: int, snapshot_resolution: int, max_snapshots: int,
        additional_options: dict = None
    ):
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
        """FrameBase: Frame instance of current business engine."""
        pass

    @property
    @abstractmethod
    def snapshots(self) -> SnapshotList:
        """SnapshotList: Snapshot list of current frame, this is used to expose querying interface for outside."""
        pass

    def frame_index(self, tick: int) -> int:
        """Helper method for child class, used to get index of frame in snapshot list for specified tick.

        Args:
            tick (int): Tick to calculate frame index.

        Returns:
            int: Frame index in snapshot list of specified tick.
        """
        return tick_to_frame_index(self._start_tick, tick, self._snapshot_resolution)

    def calc_max_snapshots(self) -> int:
        """Helper method to calculate total snapshot should be in snapshot list with parameters passed via constructor.

        NOTE:
            This method will return max number that can contains all the frame state to the end.
            You can use a small size to hold states, when hit the limitation, oldest one will be overwrote.

        Returns:
            int: Max snapshot number for current configuration.
        """
        return self._max_snapshots if self._max_snapshots is not None \
            else total_frames(self._start_tick, self._max_tick, self._snapshot_resolution)

    def update_config_root_path(self, business_engine_file_path: str):
        """Helper method used to update the config path with business engine path if you
        follow the way to load configuration file as built-in scenarios.

        This method assuming that all the configuration (topologies) is under their scenario folder,
        and named as topologies, each topology is one folder.

        NOTE:
            You can use your own way to place the configuration files, and ignore this.

        Examples:

            .. code-block:: python

                # Define a business engine.
                class MyBusinessEngine(AbsBusinessEngine):
                    def __init__(self, *args, **kwargs):
                        super().__init__("my_be", *args, **kwargs)

                        # Use __file__ as parameter.
                        self.update_config_root_path(__file__)

        Args:
            business_engine_file_path(str): Full path of real business engine file.
        """
        if self._topology:
            path = Path(self._topology)

            if path.exists() and path.is_dir():
                # if topology is a existing path, then use it as config root path
                self._config_path = self._topology
            else:
                be_file_path = os.path.split(os.path.realpath(business_engine_file_path))[0]
                self._config_path = os.path.join(be_file_path, "topologies", self._topology)

    @abstractmethod
    def step(self, tick: int):
        """Method that is called at each tick, usually used to trigger business logic at current tick.

        Args:
            tick (int): Current tick from simulator.
        """
        pass

    @property
    def configs(self) -> dict:
        """dict: Configurations of this business engine."""
        pass

    @abstractmethod
    def reset(self):
        """Reset states business engine."""
        pass

    def post_step(self, tick: int) -> bool:
        """This method will be called at the end of each tick, used to post-process for each tick,
        for complex business logic with many events, it maybe not easy to determine
        if stop the scenario at the middle of tick, so this method is used to avoid this.

        Args:
            tick (int): Current tick.

        Returns:
            bool: If simulator should stop simulation at current tick.
        """
        return False

    def get_node_mapping(self) -> dict:
        """Get mapping for nodes, like index->name or index->id, may different for scenarios.

        Returns:
            dict: Key is node index, value is decided by scenario, usually is name or id.
        """
        return {}

    def get_event_payload_detail(self) -> dict:
        """Get payload keys for all kinds of event.

        For the performance of the simulator, some event payload has no corresponding Python object.
        This mapping is provided for your convenience in such case.

        Returns:
            dict: Key is the event type in string format, value is a list of available keys.
        """
        return {}

    def get_metrics(self) -> dict:
        """Get statistics information, may different for scenarios.

        Returns:
            dict: Dictionary about metrics, content and format determined by business engine.
        """
        return {}

    def dump(self, folder: str):
        """Dump something from business engine.

        Args:
            folder (str): Folder to place dumped files.
        """
        pass
