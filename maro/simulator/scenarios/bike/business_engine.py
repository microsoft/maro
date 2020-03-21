
import csv
import math
import os
from enum import IntEnum
from typing import Dict, List

from yaml import safe_load

from maro.simulator.event_buffer import DECISION_EVENT, Event, EventBuffer
from maro.simulator.graph import Graph, SnapshotList
from maro.simulator.scenarios import AbsBusinessEngine

from .cell import Cell
from .common import (Action, BikeReturnPayload, BikeTransferPayload,
                     DecisionEvent, Trip)
from .trip_reader import BikeTripReader
from .decision_strategy import BikeDecisionStrategy
from .resource_builder import build

class BikeEventType(IntEnum):
    """Events we need to handled to process trip logic"""
    TripRequirement = 1    # a user need a bike
    BikeReturn = 2         # user return the bike at target cell
    BikeTransfermation = 3 # transfer bikes from a cell to another

class BikeBusinessEngine(AbsBusinessEngine):
    def __init__(self, event_buffer: EventBuffer, config_path: str, max_tick: int, tick_units: int):
        super().__init__(event_buffer, config_path, tick_units)

        self._decision_strategy = None
        self._max_tick = max_tick
        self._cells = []
        self._cell_map = {} # TODO: can be removed after we have actually have cell

        config_path = os.path.join(config_path, "config.yml")

        self._conf = None

        with open(config_path) as fp:
            self._conf = safe_load(fp)

        self._init_graph()
        self._init_data_reader()
        
        self._snapshots = SnapshotList(self._graph, max_tick)

        self._reg_event()
        self._init_decision_strategy()

    @property
    def graph(self) -> Graph:
        """Graph: Graph of current business engine
        """
        return self._graph

    @property
    def snapshots(self) -> SnapshotList:
        """SnapshotList: Snapshot list of current graph"""
        return self._snapshots

    def step(self, tick: int, internal_tick: int):
        """Used to process events at specified tick, usually this is called by Env at each internal tick

        Args:
            tick (int): tick to process
            internal_tick (int): internal tick (in minute) to process trip data
        """

        # get trip requiremnts for current internal tick (in minute)
        trips = self._data_reader.get_trips(internal_tick)
        
        # generate events to process
        for trip in trips:
            trip_evt = self._event_buffer.gen_atom_event(internal_tick, BikeEventType.TripRequirement, payload=trip)

            self._event_buffer.insert_event(trip_evt)

        cells_need_decision = self._decision_strategy.get_cells_need_decision(tick ,internal_tick)

        # the env will take snapshot for use when we need an action, so we do not need to take action here
        for cell_idx in cells_need_decision:
            # we use tick (in hour) here, not internal tick, as agent do not need to known this
            decision_payload = DecisionEvent(cell_idx, tick, self._decision_strategy.action_scope)
            decision_evt  = self._event_buffer.gen_cascade_event(unit_tick, DECISION_EVENT, deciton_payload)

            self._event_buffer.insert_event(decision_evt)

    @property
    def configs(self) -> dict:
        """object: Configurations of this business engine"""
        return self._conf

    def rewards(self, actions) -> float:
        """Calculate rewards based on actions

        Args:
            actions(list): Action(s) from agent

        Returns:
            float: reward based on actions
        """
        return []

    def reset(self):
        """Reset business engine"""
        # clear value in snapshots
        self._snapshots.reset()

        # clear value in current graph
        self._graph.reset()

        # prepare data pointer to beginning
        self._data_reader.reset()

        # reset cell to initial value
        for cell in self._cells:
            cell.reset()

    def get_node_name_mapping(self) -> Dict[str, Dict]:
        """Get node name mappings related with this environment

        Returns:
            Dict[str, Dict]: Node name to index mapping dictionary
        """
        return {}

    def get_agent_idx_list(self) -> List[int]:
        """Get port index list related with this environment

        Returns:
            List[int]: list of port index
        """
        return [i for i in range(len(self._cells))]

    def post_step(self, tick: int, unit_tick: int):
        """Post-process at specified tick

        Args:
            tick (int): tick to process

        """
        if (unit_tick + 1) % self._tick_units == 0:
            # take a snapshot at the end of tick
            self._snapshots.insert_snapshot(self._graph, tick)

            # last unit tick of current tick
            # we will reset some field
            for cell in self._cells:
                cell.shortage = 0
                cell.trip_requirement = 0
                cell.unknow_gendors = 0
                cell.males = 0
                cell.females = 0
                cell.weekday = 0
                cell.customer = 0
                cell.subscriptor = 0

    def _init_graph(self):
        rows = []
        with open(self._conf["cell_file"]) as fp:
            reader = csv.reader(fp)

            for l in reader:
                rows.append(l)

        self._graph = build(len(rows))
  
        for i, r in enumerate(rows):
            if len(r) == 0:
                break

            cell = Cell(i, int(r[0]), int(r[2]), int(r[3]), self._graph)

            self._cells.append(cell)
            self._cell_map[int(r[0])] = i

    def _init_data_reader(self):
        self._data_reader = BikeTripReader(self._conf["trip_file"], 
                                            self._conf["start_datetime"], 
                                            self._max_tick, self._cell_map)


    def _init_decision_strategy(self):
        self._decision_strategy = BikeDecisionStrategy(self._cells, self._conf["decision"])

    def _reg_event(self):
        self._event_buffer.register_event_handler(BikeEventType.TripRequirement, self._on_trip_gen)
        self._event_buffer.register_event_handler(BikeEventType.BikeReturn, self._on_bike_return)
        
        # decision event, predefined in event buffer
        self._event_buffer.register_event_handler(DECISION_EVENT, self._on_action_recieved)
        self._event_buffer.register_event_handler(BikeEventType.BikeTransfermation , self._on_bike_recieved)

    def _on_trip_gen(self, evt: Event):
        """On order generated:
        1. try to remove a bike from inventory
        2. update shortage, gendor, usertype and weekday statistics states"""

        order: Order = evt.payload
        cell_idx: int = order.from_cell
        cell: Cell = self._cells[cell_idx]
        cell_bikes = cell.bikes

        # update order count
        cell.trip_requirement += 1
        
        if cell_bikes <= 0:
            # shortage
            cell.shortage += 1
        else:
            cell.bikes = cell_bikes - 1

            # TODO: update gender, weekday and usertype 
            cell.update_gendor(order.gendor)
            cell.update_usertype(order.usertype)
            cell.weekday = order.weekday

            # generate a bike return event by end tick
            return_payload = BikeReturnPayload(order.from_cell, order.to_cell)
            bike_return_evt = self._event_buffer.gen_atom_event(order.end_tick, BikeEventType.BikeReturn, payload=return_payload)

            self._event_buffer.insert_event(bike_return_evt)

    def _on_bike_return(self, evt: Event):
        payload: BikeReturnPayload = evt.payload
        target_cell: Cell = self._cells[payload.to_cell]


        # TODO: what about more than capacity?
        target_cell.bikes += 1

    def _on_action_recieved(self, evt: Event):
        action: Action  = None
        
        for action in evt.payload:
            cell: Cell = self._cells[action.from_cell]

            executed_number = min(cell.bikes, action.number)
            cell.bikes -= executed_number

            payload = BikeTransferPaylod(action.from_cell, action.to_cell, action.number)

            # TODO: apply random transfer ticks
            transfer_evt = self._event_buffer.gen_atom_event(evt.tick + 10, BikeEventType.BikeTransfermation, payload)
            
            self._event_buffer.insert_event(transfer_evt)

    def _on_bike_recieved(self, evt: Event):
        payload: BikeTransferPaylod = evt.payload
        cell: Cell = self._cells[payload.to_cell]

        # TODO: what about if out of capacity
        cell.bikes += payload.number
