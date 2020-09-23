# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import os

from math import floor
from importlib import import_module
from inspect import getmembers, isclass
from typing import Any, Dict, List, Tuple
from collections import Iterable

from maro.event_buffer import DECISION_EVENT, EventBuffer, EventState
from maro.backends.frame import FrameBase, SnapshotList
from maro.utils.exception.simulator_exception import BusinessEngineNotFoundError

from .utils.common import tick_to_frame_index
from .utils import seed as sim_seed
from .abs_core import AbsEnv, DecisionMode
from .scenarios.abs_business_engine import AbsBusinessEngine


class Env(AbsEnv):
    """Default environment

    Args:
        scenario (str): scenario name under maro/sim/scenarios folder
        topology (str): topology name under specified scenario folder, if this point to a existing folder, then it will use this as topology for built-in scenario
        start_tick (int): start tick of the scenario, usually used for pre-processed data streaming
        durations (int): duration ticks of this environment from start_tick
        snapshot_resolution (int): how many ticks will take a snapshot
        max_snapshots(int): max in-memory snapshot number, default None means keep all snapshots in memory, when taking a snapshot, if it reaches this limitation, oldest one will be overwrote.
        business_engine_cls : class of business engine, if specified, then use it to construct be instance, or will search internal by scenario
        options (dict): additional parameters passed to business engine

    """

    def __init__(self, scenario: str = None, topology: str = None,
                 start_tick: int = 0, durations: int = 100, snapshot_resolution: int = 1,  max_snapshots: int = None,
                 decision_mode: DecisionMode = DecisionMode.Sequential,
                 business_engine_cls: type = None,
                 options: dict = {}):
        super().__init__(scenario, topology, start_tick, durations,
                         snapshot_resolution, max_snapshots, decision_mode, business_engine_cls, options)

        self._name = f'{self._scenario}:{self._topology}' if business_engine_cls is None else business_engine_cls.__name__
        self._business_engine: AbsBusinessEngine = None

        self._event_buffer = EventBuffer()

        # generator to push the simulator moving on
        self._simulate_generator = self._simulate()

        # initialize business
        self._init_business_engine()

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

        try:
            reward, decision_event, _is_done = self._simulate_generator.send(
                action)
        except StopIteration:
            return None, None, True

        return reward, decision_event, _is_done

    def dump(self):
        """Dump environment for restore

        NOTE:
            not implemented
        """
        return

    def reset(self):
        """Reset environment"""
        # . reset self
        self._tick = self._start_tick

        self._simulate_generator.close()
        self._simulate_generator = self._simulate()

        # . reset event buffer
        self._event_buffer.reset()

        # . ask business engine reset itself
        self._business_engine.reset()

    @property
    def configs(self) -> dict:
        """object: Configurations of current environment"""
        return self._business_engine.configs

    @property
    def summary(self) -> dict:
        """Summary about current simulator, include node details, and mappings
        
        NOTE: this is provided by scenario, so may have different format and content
        """
        return {
            "node_mapping": self._business_engine.get_node_mapping(),
            "node_detail": self.current_frame.get_node_info()
        }
        

    @property
    def name(self) -> str:
        """str: Name of current environment"""
        return self._name

    @property
    def current_frame(self) -> FrameBase:
        """Frame: Frame of current environment"""
        return self._business_engine.frame

    @property
    def tick(self) -> int:
        """int: Current tick of environment"""
        return self._tick

    @property
    def frame_index(self) -> int:
        """int: frame index in snapshot list for current tick"""
        return tick_to_frame_index(self._start_tick, self._tick, self._snapshot_resolution)

    @property
    def snapshot_list(self) -> SnapshotList:
        """SnapshotList: Current snapshot list

        a snapshot list contains all the snapshots of frame at each tick
        """
        return self._business_engine.snapshots

    @property
    def agent_idx_list(self) -> List[int]:
        """List[int]: Agent index list that related to this environment"""
        return self._business_engine.get_agent_idx_list()

    def set_seed(self, seed: int):
        """Set random seed used by simulator.
        
        NOTE: this will not set seed for python random or other packages' seed, such as numpy.
        
        Args:
            seed (int): 
        """

        if seed is not None:
            sim_seed(seed)

    @property
    def metrics(self) -> dict:
        """Some statistics information provided by business engine
        
        Returns:
            dict: dictionary of metrics, content and format is determined by business engine
        """

        return self._business_engine.get_metrics()

    def get_finished_events(self):
        """List[Event]: All events finished so far
        """
        return self._event_buffer.get_finished_events()

    def get_pending_events(self, tick):
        """
        Pending events at certain tick

        Args:
            tick (int): Specified tick
        """
        return self._event_buffer.get_pending_events(tick)

    def _init_business_engine(self):
        """Initialize business engine object.

        NOTE:
        1. internal scenarios will always under "maro/simulator/scenarios" folder
        2. external scenarios, we access the business engine class to create instance 
        """
        max_tick = self._start_tick + self._durations

        if self._business_engine_cls is not None:
            business_class = self._business_engine_cls
        else:
            # combine the business engine import path
            business_class_path = f'maro.simulator.scenarios.{self._scenario}.business_engine'

            # load the module to find business engine for that scenario
            business_module = import_module(business_class_path)

            business_class = None

            for _, obj in getmembers(business_module, isclass):
                if issubclass(obj, AbsBusinessEngine) and obj != AbsBusinessEngine:
                    # we find it
                    business_class = obj

                    break

            if business_class is None:
                raise BusinessEngineNotFoundError()

        self._business_engine = business_class(event_buffer=self._event_buffer, 
                                            topology=self._topology,
                                            start_tick=self._start_tick, 
                                            max_tick=max_tick, 
                                            snapshot_resolution=self._snapshot_resolution, 
                                            max_snapshots=self._max_snapshots,
                                            additional_options=self._additional_options)

    def _simulate(self):
        """
        this is the generator to wrap each episode process
        """
        is_end_tick = False

        while True:
            # ask business engine to do thing for this tick, such as gen and push events
            # we do not push events now
            self._business_engine.step(self._tick)

            while True:
                # we keep process all the events, until no more any events
                pending_events = self._event_buffer.execute(self._tick)

                # processing pending events
                pending_event_length: int = len(pending_events)

                if pending_event_length == 0:
                    # we have processed all the event of current tick, lets go for next tick
                    break

                # insert snapshot before each action
                self._business_engine.frame.take_snapshot(self.frame_index)

                decision_events = []

                # append source event id to decision events, to support sequential action in joint mode
                for evt in pending_events:
                    payload = evt.payload

                    payload.source_event_id = evt.id

                    decision_events.append(payload)


                decision_events = decision_events[0] if self._decision_mode == DecisionMode.Sequential else decision_events

                # yield current state first, and waiting for action
                actions = yield self._business_engine.get_metrics(), decision_events, False

                if actions is None:
                    actions = [] # make business engine easy to work

                if actions is not None and not isinstance(actions, Iterable):
                    actions = [actions]

                # generate a new atom event first
                action_event = self._event_buffer.gen_atom_event(self._tick, DECISION_EVENT, actions)

                # 3. we just append the action into sub event of first pending cascade event
                pending_events[0].state = EventState.EXECUTING
                pending_events[0].immediate_event_list.append(action_event)

                # TODO: support get reward after action complete here, via using event_buffer.execute

                if self._decision_mode == DecisionMode.Joint:
                    # for joint event, we will disable following cascade event
                    
                    # we expect that first action contains a src_event_id to support joint event with sequential action
                    action_related_event_id = None if len(actions) == 1 else getattr(actions[0], "src_event_id", None)

                    # if first action have decision event attached, then means support sequential action
                    is_support_seq_action = action_related_event_id is not None

                    if is_support_seq_action:
                        for i in range(1, pending_event_length):
                            if pending_events[i].id == actions[0].src_event_id:
                                pending_events[i].state = EventState.FINISHED
                    else:
                        for i in range(1, pending_event_length):
                            pending_events[i].state = EventState.FINISHED

            # check if we should end simulation
            is_end_tick = self._business_engine.post_step(self._tick) == True

            if is_end_tick:
                break

            self._tick += 1

        # make sure we have no missing data
        if (self._tick + 1) % self._snapshot_resolution != 0:
            self._business_engine.frame.take_snapshot(self.frame_index)

        # the end
        yield self._business_engine.get_metrics(), None, True
