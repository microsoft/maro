# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import os

from typing import Dict, Tuple, List, Any
from importlib import import_module
from inspect import isclass, getmembers

from .graph import Graph
from .graph import SnapshotList

from .abs_core import AbsEnv, DecisionMode
from .event_buffer import EventBuffer, EventState, DECISION_EVENT
from .scenarios import AbsBusinessEngine

class BusinessEngineNotFoundError(Exception):
    """Cannot load and initialize related business engine class"""

    def __init__(self, msg):
        self.message = msg


class BusinessInitializationError(Exception):
    """Business engine not initialized before using"""
    def __init__(self, msg):
        self.message = msg


class Env(AbsEnv):
    """Default environment

    Note:
        Default environment assuming that the all the all the scenarios are put into maro/simulator/scenarios folder,
        so user cannot use this environment with customized scenario without source code.

        Also it will try to load business_engine.py under each scenario folder (like ecr), so each scenario folder should
        have their own business_engine.py file.

    Args:
        scenario (str): scenario name under maro.simulator/scenarios folder
        topology (topology): topology name under specified scenario folder
        max_tick (int): max tick of this environment, default is 100
        decision_mode (DecisionMode): decision mode that specified interactive mode with agent

    """
    def __init__(self, scenario: str, topology: str, max_tick: int = 100, decision_mode=DecisionMode.Sequential):
        assert max_tick > 0

        super().__init__(scenario, topology, max_tick, decision_mode)

        self._scenario = scenario
        self._topology = topology
        self._max_tick = max_tick
        self._tick = 0
        self._name = f'{self._scenario}:{self._topology}'
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

            - decision_event for sequential decision mode, or a list of decision_event (the pending event can be any object,
              like DecisionEvent for ECR scenario)"

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
        self._tick = 0

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
    def node_name_mapping(self) -> Dict[str, dict]:
        """Dict[str, List]: Resource node name mapping that configured for current environment"""
        return self._business_engine.get_node_name_mapping()

    @property
    def name(self) -> str:
        """str: Name of current environment"""
        return self._name

    @property
    def current_graph(self) -> Graph:
        """Graph: Graph of current environment"""
        return self._business_engine.graph

    @property
    def tick(self) -> int:
        """int: Current tick of environment"""
        return self._tick

    @property
    def snapshot_list(self) -> SnapshotList:
        """SnapshotList: Current snapshot list

        a snapshot list contains all the snapshots of graph at each tick
        """
        return self._business_engine.snapshots

    @property
    def agent_idx_list(self) -> List[int]:
        """List[int]: Agent index list that related to this environment"""
        return self._business_engine.get_agent_idx_list()

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

        NOTE: we are assuming that each scenario folder contains an business_engine.py (maybe from config later).
        then load and initialize it from that file
        """
        # combine the business engine import path
        business_class_path = f'maro.simulator.scenarios.{self._scenario}.business_engine'

        # load the module to find business engine for that scenario
        business_module = import_module(business_class_path)

        business_class = None

        for name, obj in getmembers(business_module, isclass):
            if issubclass(obj, AbsBusinessEngine) and obj != AbsBusinessEngine:
                # we find it
                business_class = obj

                break

        if business_class is None:
            raise BusinessEngineNotFoundError(
                "Business engine not find for scenario ecr")

        topology_path = os.path.join(os.path.split(os.path.realpath(__file__))[
                                     0], "scenarios", self._scenario, "topologies", self._topology)

        self._business_engine = business_class(
            self._event_buffer, topology_path, self._max_tick)

        # check if it meet our requirement
        if self._business_engine.graph is None:
            raise BusinessInitializationError(
                "graph of business engine is None")

    def _simulate(self):
        """
        this is the generator to wrap each episode process
        """
        rewards = None  # default value of reward

        while self._tick < self._max_tick:
            # ask business engine to do thing for this tick, such as gen and push events
            # we do not push events now
            self._business_engine.step(self._tick)

            while True:
                # we keep process all the events, util no more any events
                pending_events = self._event_buffer.execute(self._tick)

                # processing pending events
                pending_event_length: int = len(pending_events)

                if pending_event_length == 0:
                    # we have processed all the event of current tick, lets go for next tick
                    break

                # insert snapshot before each action
                self._business_engine.snapshots.insert_snapshot(
                    self.current_graph, self.tick)

                decision_events = [evt.payload for evt in pending_events]

                decision_events = decision_events[0] if self._decision_mode == DecisionMode.Sequential else decision_events

                # yield current state first, and waiting for action
                actions = yield rewards, decision_events, False

                if actions is not None and type(actions) is not list:
                    actions = [actions]

                # calculate rewards
                rewards = self._business_engine.rewards(actions)

                # unpack reward there is only one
                if len(rewards) == 1:
                    rewards = rewards[0]

                # generate a new atom event first
                action_event = self._event_buffer.gen_atom_event(
                    self._tick, DECISION_EVENT, actions)

                # 3. we just append the action into sub event of first pending cascade event
                pending_events[0].state = EventState.EXECUTING
                pending_events[0].immediate_event_list.append(action_event)

                if self._decision_mode == DecisionMode.Joint:
                    # for joint event, we will disable following cascade event
                    for i in range(1, pending_event_length):
                        pending_events[i].state = EventState.FINISHED

            self._business_engine.post_step(self._tick)
            self._tick += 1

        # reset the tick to avoid add one more time at the end of loop
        self._tick = self._max_tick - 1

        # the end
        yield None, None, True