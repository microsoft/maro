# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from importlib import import_module
from inspect import getmembers, isclass
from typing import Generator, List, Optional, Tuple, Union, cast

from maro.backends.frame import FrameBase, SnapshotList
from maro.data_lib.dump_csv_converter import DumpConverter
from maro.event_buffer import ActualEvent, CascadeEvent, EventBuffer, EventState
from maro.streamit import streamit
from maro.utils.exception.simulator_exception import BusinessEngineNotFoundError

from ..common import BaseAction, BaseDecisionEvent
from .abs_core import AbsEnv, DecisionMode
from .scenarios.abs_business_engine import AbsBusinessEngine
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
        self,
        scenario: str = None,
        topology: str = None,
        start_tick: int = 0,
        durations: int = 100,
        snapshot_resolution: int = 1,
        max_snapshots: int = None,
        decision_mode: DecisionMode = DecisionMode.Sequential,
        business_engine_cls: type = None,
        disable_finished_events: bool = False,
        record_finished_events: bool = False,
        record_file_path: str = None,
        options: Optional[dict] = None,
    ) -> None:
        super().__init__(
            scenario,
            topology,
            start_tick,
            durations,
            snapshot_resolution,
            max_snapshots,
            decision_mode,
            business_engine_cls,
            disable_finished_events,
            options if options is not None else {},
        )

        self._name = (
            f"{self._scenario}:{self._topology}" if business_engine_cls is None else business_engine_cls.__name__
        )

        self._event_buffer = EventBuffer(disable_finished_events, record_finished_events, record_file_path)

        # decision_payloads array for dump.
        self._decision_payloads = []

        # The generator used to push the simulator forward.
        self._simulate_generator = self._simulate()

        # Initialize the business engine.
        self._init_business_engine()

        if "enable-dump-snapshot" in self._additional_options:
            parent_path = self._additional_options["enable-dump-snapshot"]
            self._converter = DumpConverter(parent_path, self._business_engine.scenario_name)
            self._converter.reset_folder_path()

        self._streamit_episode = 0

    def step(
        self,
        action: Union[BaseAction, List[BaseAction], None] = None,
    ) -> Tuple[Optional[dict], Union[BaseDecisionEvent, List[BaseDecisionEvent], None], bool]:
        """Push the environment to next step with action.

        Under Sequential mode:
            - If `action` is None, an empty list will be assigned to the decision event.
            - Otherwise, the action(s) will be assigned to the decision event.

        Under Joint mode:
            - If `action` is None, no actions will be assigned to any decision event.
            - If `action` is a single action, it will be assigned to the first decision event.
            - If `action` is a list, actions are assigned to each decision event in order. If the number of actions
                is less than the number of decision events, extra decision events will not be assigned actions. If
                the number of actions if larger than the number of decision events, extra actions will be ignored.
            If you want to assign multiple actions to specific event(s), please explicitly pass a list of list. For
            example:

            ```
            env.step(action=[[a1, a2], a3, [a4, a5]])
            ```

            Will assign `a1` & `a2` to the first decision event, `a3` to the second decision event, and `a4` & `a5`
            to the third decision event.

            Particularly, if you only want to assign multiple actions to the first decision event, please
            pass `[[a1, a2, ..., an]]` (a list of one list) instead of `[a1, a2, ..., an]` (an 1D list of n elements),
            since the latter one will assign the n actions to the first n decision events.

        Args:
            action (Union[BaseAction, List[BaseAction], None]): Action(s) from agent.

        Returns:
            tuple: a tuple of (metrics, decision event, is_done).
        """
        try:
            metrics, decision_payloads, _is_done = self._simulate_generator.send(action)
        except StopIteration:
            return None, None, True

        return metrics, decision_payloads, _is_done

    def dump(self) -> None:
        """Dump environment for restore.

        NOTE:
            Not implemented.
        """
        return

    def reset(self, keep_seed: bool = False) -> None:
        """Reset environment.

        Args:
            keep_seed (bool): Reset the random seed to the generate the same data sequence or not. Defaults to False.
        """
        self._tick = self._start_tick

        self._simulate_generator.close()
        self._simulate_generator = self._simulate()

        self._event_buffer.reset()

        if "enable-dump-snapshot" in self._additional_options and self._business_engine.frame is not None:
            dump_folder = self._converter.get_new_snapshot_folder()

            self._business_engine.frame.dump(dump_folder)
            self._converter.start_processing(self.configs)
            self._converter.dump_descsion_events(
                self._decision_payloads,
                self._start_tick,
                self._snapshot_resolution,
            )
            self._business_engine.dump(dump_folder)

        self._decision_payloads.clear()

        self._business_engine.reset(keep_seed)

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

    def set_seed(self, seed: int) -> None:
        """Set random seed used by simulator.

        NOTE:
            This will not set seed for Python random or other packages' seed, such as NumPy.

        Args:
            seed (int): Seed to set.
        """
        assert seed is not None and isinstance(seed, int)
        self._business_engine.set_seed(seed)

    @property
    def metrics(self) -> dict:
        """Some statistics information provided by business engine.

        Returns:
            dict: Dictionary of metrics, content and format is determined by business engine.
        """

        return self._business_engine.get_metrics()

    def get_finished_events(self) -> List[ActualEvent]:
        """List[Event]: All events finished so far."""
        return self._event_buffer.get_finished_events()

    def get_pending_events(self, tick) -> List[ActualEvent]:
        """Pending events at certain tick.

        Args:
            tick (int): Specified tick to query.
        """
        return self._event_buffer.get_pending_events(tick)

    def get_ticks_frame_index_mapping(self) -> dict:
        """Helper method to get current available ticks to related frame index mapping.

        Returns:
            dict: Dictionary of avaliable tick to frame index, it would be 1 to N mapping if the resolution is not 1.
        """
        return self._business_engine.get_ticks_frame_index_mapping()

    def _init_business_engine(self) -> None:
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
            business_class_path = f"maro.simulator.scenarios.{self._scenario}.business_engine"

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

        self._business_engine: AbsBusinessEngine = business_class(
            event_buffer=self._event_buffer,
            topology=self._topology,
            start_tick=self._start_tick,
            max_tick=max_tick,
            snapshot_resolution=self._snapshot_resolution,
            max_snapshots=self._max_snapshots,
            additional_options=self._additional_options,
        )

    def _assign_action(
        self,
        action: Union[BaseAction, List[BaseAction], None],
        decision_event: CascadeEvent,
    ) -> None:
        decision_event.state = EventState.EXECUTING

        if action is None:
            actions = []
        elif not isinstance(action, list):
            actions = [action]
        else:
            actions = action

        decision_event.add_immediate_event(self._event_buffer.gen_action_event(self._tick, actions), is_head=True)

    def _simulate(
        self,
    ) -> Generator[
        Tuple[dict, Union[BaseDecisionEvent, List[BaseDecisionEvent]], bool],
        Union[BaseAction, List[BaseAction], None],
        None,
    ]:
        """This is the generator to wrap each episode process."""
        self._streamit_episode += 1

        streamit.episode(self._streamit_episode)

        while True:
            # Ask business engine to do thing for this tick, such as generating and pushing events.
            # We do not push events now.
            streamit.tick(self._tick)

            self._business_engine.step(self._tick)

            while True:
                # Keep processing events, until no more events in this tick.
                pending_events = cast(List[CascadeEvent], self._event_buffer.execute(self._tick))

                if len(pending_events) == 0:
                    # We have processed all the event of current tick, lets go for next tick.
                    break

                # Insert snapshot before each action.
                self._business_engine.frame.take_snapshot(self.frame_index)

                # Append source event id to decision events, to support sequential action in joint mode.
                decision_payloads = [event.payload for event in pending_events]

                if self._decision_mode == DecisionMode.Sequential:
                    self._decision_payloads.append(decision_payloads[0])
                    action = yield self._business_engine.get_metrics(), decision_payloads[0], False
                    self._assign_action(action, pending_events[0])
                else:
                    self._decision_payloads += decision_payloads
                    actions = yield self._business_engine.get_metrics(), decision_payloads, False
                    if actions is None:
                        actions = []
                    assert isinstance(actions, list)

                    for action, event in zip(actions, pending_events):
                        self._assign_action(action, event)

                    if self._decision_mode == DecisionMode.Joint:
                        for event in pending_events[len(actions) :]:
                            event.state = EventState.FINISHED

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
