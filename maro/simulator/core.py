# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import Iterable
from importlib import import_module
from inspect import getmembers, isclass
from typing import List

from maro.backends.frame import FrameBase, SnapshotList
from maro.data_lib.dump_csv_converter import DumpConverter
from maro.event_buffer import EventBuffer, EventState
from maro.streamit import streamit
from maro.utils.exception.simulator_exception import BusinessEngineNotFoundError

from .abs_core import AbsEnv, DecisionMode
from .scenarios.abs_business_engine import AbsBusinessEngine
from .utils import seed as sim_seed
from .utils.common import tick_to_frame_index


class Env(AbsEnv):
    """Default environment implementation using generator.

    Args:
        scenario (str): Scenario name under maro/simulator/scenarios folder.
        topology (str): Topology name under specified scenario folder.
            If it points to an existing folder, the corresponding topology will be used for the built-in scenario.
        start_tick (int): Start tick of the scenario, usually used for pre-processed data streaming.
        durations (int): Duration ticks of this environment from start_tick.
        snapshot_resolution (int): How many ticks will take a snapshot.
        max_snapshots(int): Max in-memory snapshot number.
            When the number of dumped snapshots reached the limitation, oldest one will be overwrote by new one.
            None means keeping all snapshots in memory. Defaults to None.
        business_engine_cls (type): Class of business engine. If specified, use it to construct the be instance,
            or search internally by scenario.
        disable_finished_events (bool): Disable finished events list, with this set to True, EventBuffer will
            re-use finished event object, this reduce event object number.
        record_finished_events (bool): If record finished events into csv file, default is False.
        record_file_path (str): Where to save the recording file, only work if record_finished_events is True.
        options (dict): Additional parameters passed to business engine.
    """

    def __init__(
        self, scenario: str = None, topology: str = None,
        start_tick: int = 0, durations: int = 100, snapshot_resolution: int = 1, max_snapshots: int = None,
        decision_mode: DecisionMode = DecisionMode.Sequential,
        business_engine_cls: type = None, disable_finished_events: bool = False,
        record_finished_events: bool = False,
        record_file_path: str = None,
        options: dict = {}
    ):
        super().__init__(
            scenario, topology, start_tick, durations,
            snapshot_resolution, max_snapshots, decision_mode, business_engine_cls,
            disable_finished_events, options
        )

        self._name = f'{self._scenario}:{self._topology}' if business_engine_cls is None \
            else business_engine_cls.__name__
        self._business_engine: AbsBusinessEngine = None

        self._event_buffer = EventBuffer(disable_finished_events, record_finished_events, record_file_path)

        # decision_events array for dump.
        self._decision_events = []

        # The generator used to push the simulator forward.
        self._simulate_generator = self._simulate()

        # Initialize the business engine.
        self._init_business_engine()

        if "enable-dump-snapshot" in self._additional_options:
            parent_path = self._additional_options["enable-dump-snapshot"]
            self._converter = DumpConverter(parent_path, self._business_engine._scenario_name)
            self._converter.reset_folder_path()

        self._streamit_episode = 0

    def step(self, action):
        """Push the environment to next step with action.

        Args:
            action (Action): Action(s) from agent.

        Returns:
            tuple: a tuple of (metrics, decision event, is_done).
        """
        try:
            metrics, decision_event, _is_done = self._simulate_generator.send(action)
        except StopIteration:
            return None, None, True

        return metrics, decision_event, _is_done

    def dump(self):
        """Dump environment for restore.

        NOTE:
            Not implemented.
        """
        return

    def reset(self):
        """Reset environment."""
        self._tick = self._start_tick

        self._simulate_generator.close()
        self._simulate_generator = self._simulate()

        self._event_buffer.reset()

        if ("enable-dump-snapshot" in self._additional_options) and (self._business_engine._frame is not None):
            dump_folder = self._converter.get_new_snapshot_folder()

            self._business_engine._frame.dump(dump_folder)
            self._converter.start_processing(self.configs)
            self._converter.dump_descsion_events(self._decision_events, self._start_tick, self._snapshot_resolution)
            self._business_engine.dump(dump_folder)

        self._decision_events.clear()

        self._business_engine.reset()

    @property
    def configs(self) -> dict:
        """dict: Configurations of current environment."""
        return self._business_engine.configs

    @property
    def summary(self) -> dict:
        """dict: Summary about current simulator, including node details and mappings."""
        return {
            "node_mapping": self._business_engine.get_node_mapping(),
            "node_detail": self.current_frame.get_node_info(),
            "event_payload": self._business_engine.get_event_payload_detail(),
        }

    @property
    def name(self) -> str:
        """str: Name of current environment."""
        return self._name

    @property
    def current_frame(self) -> FrameBase:
        """Frame: Frame of current environment."""
        return self._business_engine.frame

    @property
    def tick(self) -> int:
        """int: Current tick of environment."""
        return self._tick

    @property
    def frame_index(self) -> int:
        """int: Frame index in snapshot list for current tick."""
        return tick_to_frame_index(self._start_tick, self._tick, self._snapshot_resolution)

    @property
    def snapshot_list(self) -> SnapshotList:
        """SnapshotList: A snapshot list containing all the snapshots of frame at each dump point.

        NOTE: Due to different environment configurations, the resolution of the snapshot may be different.
        """
        return self._business_engine.snapshots

    @property
    def agent_idx_list(self) -> List[int]:
        """List[int]: Agent index list that related to this environment."""
        return self._business_engine.get_agent_idx_list()

    def set_seed(self, seed: int):
        """Set random seed used by simulator.

        NOTE:
            This will not set seed for Python random or other packages' seed, such as NumPy.

        Args:
            seed (int): Seed to set.
        """

        if seed is not None:
            sim_seed(seed)

    @property
    def metrics(self) -> dict:
        """Some statistics information provided by business engine.

        Returns:
            dict: Dictionary of metrics, content and format is determined by business engine.
        """

        return self._business_engine.get_metrics()

    def get_finished_events(self):
        """List[Event]: All events finished so far."""
        return self._event_buffer.get_finished_events()

    def get_pending_events(self, tick):
        """Pending events at certain tick.

        Args:
            tick (int): Specified tick to query.
        """
        return self._event_buffer.get_pending_events(tick)

    def _init_business_engine(self):
        """Initialize business engine object.

        NOTE:
        1. For built-in scenarios, they will always under "maro/simulator/scenarios" folder.
        2. For external scenarios, the business engine instance is built with the loaded business engine class.
        """
        max_tick = self._start_tick + self._durations

        if self._business_engine_cls is not None:
            business_class = self._business_engine_cls
        else:
            # Combine the business engine import path.
            business_class_path = f'maro.simulator.scenarios.{self._scenario}.business_engine'

            # Load the module to find business engine for that scenario.
            business_module = import_module(business_class_path)

            business_class = None

            for _, obj in getmembers(business_module, isclass):
                if issubclass(obj, AbsBusinessEngine) and obj != AbsBusinessEngine:
                    # We find it.
                    business_class = obj

                    break

            if business_class is None:
                raise BusinessEngineNotFoundError()

        self._business_engine = business_class(
            event_buffer=self._event_buffer,
            topology=self._topology,
            start_tick=self._start_tick,
            max_tick=max_tick,
            snapshot_resolution=self._snapshot_resolution,
            max_snapshots=self._max_snapshots,
            additional_options=self._additional_options
        )

    def _simulate(self):
        """This is the generator to wrap each episode process."""
        is_end_tick = False

        self._streamit_episode += 1

        streamit.episode(self._streamit_episode)

        while True:
            # Ask business engine to do thing for this tick, such as generating and pushing events.
            # We do not push events now.
            streamit.tick(self._tick)

            self._business_engine.step(self._tick)

            while True:
                # Keep processing events, until no more events in this tick.
                pending_events = self._event_buffer.execute(self._tick)

                # Processing pending events.
                pending_event_length: int = len(pending_events)

                if pending_event_length == 0:
                    # We have processed all the event of current tick, lets go for next tick.
                    break

                # Insert snapshot before each action.
                self._business_engine.frame.take_snapshot(self.frame_index)

                # Append source event id to decision events, to support sequential action in joint mode.
                decision_events = [event.payload for event in pending_events]

                decision_events = decision_events[0] if self._decision_mode == DecisionMode.Sequential \
                    else decision_events

                # Yield current state first, and waiting for action.
                actions = yield self._business_engine.get_metrics(), decision_events, False
                # archive decision events.
                self._decision_events.append(decision_events)

                if actions is None:
                    # Make business engine easy to work.
                    actions = []

                if actions is not None and not isinstance(actions, Iterable):
                    actions = [actions]

                if self._decision_mode == DecisionMode.Sequential:
                    # Generate a new atom event first.
                    action_event = self._event_buffer.gen_action_event(self._tick, actions)

                    # NOTE: decision event always be a CascadeEvent
                    # We just append the action into sub event of first pending cascade event.
                    pending_events[0].state = EventState.EXECUTING
                    pending_events[0].add_immediate_event(action_event, is_head=True)
                else:
                    # For joint mode, we will assign actions from beginning to end.
                    # Then mark others pending events to finished if not sequential action mode.
                    for i, pending_event in enumerate(pending_events):
                        if i >= len(actions):
                            if self._decision_mode == DecisionMode.Joint:
                                # Ignore following pending events that have no action matched.
                                pending_event.state = EventState.FINISHED
                        else:
                            # Set the state as executing, so event buffer will not pop them again.
                            # Then insert the action to it.
                            action = actions[i]
                            pending_event.state = EventState.EXECUTING
                            action_event = self._event_buffer.gen_action_event(self._tick, action)

                            pending_event.add_immediate_event(action_event, is_head=True)

            # Check the end tick of the simulation to decide if we should end the simulation.
            is_end_tick = self._business_engine.post_step(self._tick)

            if is_end_tick:
                break

            self._tick += 1

        # Make sure we have no missing data.
        if (self._tick + 1) % self._snapshot_resolution != 0:
            self._business_engine.frame.take_snapshot(self.frame_index)

        # The end.
        yield self._business_engine.get_metrics(), None, True
