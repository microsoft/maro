
import csv
import os
import math
from enum import IntEnum
from typing import Dict, List

from yaml import safe_load

from maro.simulator.event_buffer import DECISION_EVENT, Event, EventBuffer
from maro.simulator.graph import Graph, SnapshotList
from maro.simulator.scenarios import AbsBusinessEngine

from .common import BikeReturnPayload, Order, DecisionEvent, Action, BikeTransferPaylod
from .data_reader import BikeDataReader
from .graph_builder import build
from .station import Station
from .abs_decisionstrategy import AbsDecisionStrategy
from .percent_decisionstrategy import BikePercentDecisionStrategy

decision_dict = {
    'percent': BikePercentDecisionStrategy
}

class BikeEventType(IntEnum):
    Order = 10
    BikeReturn = 11
    BikeTransfered = 12 # transfer bikes from a station to another

class BikeBusinessEngine(AbsBusinessEngine):
    def __init__(self, event_buffer: EventBuffer, config_path: str, max_tick: int, tick_units: int):
        super().__init__(event_buffer, config_path, tick_units)

        self._decision_strategy = None
        self._max_tick = max_tick
        self._stations = []
        self._station_map = {}

        config_path = os.path.join(config_path, "config.yml")

        self._conf = None

        with open(config_path) as fp:
            self._conf = safe_load(fp)

        self._transfer_time = int(self._conf["transfer_time"])
        self._scope_percent = float(self._conf["scope_percent"])

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

    def step(self, tick: int, unit_tick: int):
        """Used to process events at specified tick, usually this is called by Env at each tick

        Args:
            tick (int): tick to process
        """
        # print(f"************** cur tick: {unit_tick} ************************")
        orders = self._data_reader.get_orders(unit_tick)

        for order in orders:
            evt = self._event_buffer.gen_atom_event(unit_tick, BikeEventType.Order, payload=order)

            self._event_buffer.insert_event(evt)

        stations_need_decision = self._decision_strategy.get_stations_need_decision(tick ,unit_tick)

        if len(stations_need_decision) > 0:
            # the env will take snapshot for use when we need an action, so we do not need to take action here
            for station_idx in stations_need_decision:
                deciton_payload = DecisionEvent(station_idx, tick, self.action_scope)
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
        for station in self._stations:
            station.reset()

    def action_scope(self, station_idx: int) -> object:
        """Get the action scope of specified agent

        Args:
            station_idx (int): index of station

        Returns:
            object: action scope object that may different for each scenario
        """
        station: Station = self._stations[station_idx]
        
        return math.floor(station.inventory * self._scope_percent)

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
        return [i for i in range(len(self._stations))]

    def post_step(self, tick: int, unit_tick: int):
        """Post-process at specified tick

        Args:
            tick (int): tick to process

        """
        if math.fmod((unit_tick + 1), self._tick_units) == 0:
            # take a snapshot at the end of tick
            self._snapshots.insert_snapshot(self._graph, tick)

            # last unit tick of current tick
            # we will reset some field
            for station in self._stations:
                station.shortage = 0
                station.orders = 0
                station.gendor_0 = 0
                station.gendor_1 = 0
                station.gendor_2 = 0
                station.weekday = 0
                station.customer = 0
                station.subscriptor = 0

    def _init_graph(self):
        rows = []
        with open(self._conf["station_file"]) as fp:
            reader = csv.reader(fp)

            for l in reader:
                rows.append(l)

        self._graph = build(len(rows))
  
        for i, r in enumerate(rows):
            if len(r) == 0:
                break

            station = Station(i, int(r[0]), int(r[2]), int(r[3]), self._graph)

            self._stations.append(station)
            self._station_map[int(r[0])] = i

    def _init_data_reader(self):
        self._data_reader = BikeDataReader(self._conf["data_file"], self._conf["start_datetime"], self._max_tick, self._station_map)


    def _init_decision_strategy(self):
        decision_type = self._conf["decision"]["type"]

        if decision_type in decision_dict:
            self._decision_strategy = decision_dict[decision_type](self._stations, self._conf["decision"]["config"]) 
        else:
            raise "Invalid decision type"

    def _reg_event(self):
        self._event_buffer.register_event_handler(BikeEventType.Order, self._on_order_gen)
        self._event_buffer.register_event_handler(BikeEventType.BikeReturn, self._on_bike_return)
        self._event_buffer.register_event_handler(DECISION_EVENT, self._on_action_recieved)
        self._event_buffer.register_event_handler(BikeEventType.BikeTransfered, self._on_bike_recieved)

    def _on_order_gen(self, evt: Event):
        """On order generated:
        1. try to remove a bike from inventory
        2. update shortage, gendor, usertype and weekday statistics states"""

        order: Order = evt.payload
        station_idx: int = order.start_station
        station: Station = self._stations[station_idx]
        station_inventory = station.inventory

        # update order count
        station.orders += 1
        station.acc_orders += 1
        
        if station_inventory <= 0:
            # shortage
            station.shortage += 1
            station.acc_shortage += 1
        else:
            station.inventory = station_inventory - 1

            # TODO: update gender, weekday and usertype 
            station.update_gendor(order.gendor)
            station.update_usertype(order.usertype)
            station.weekday = order.weekday

            # generate a bike return event by end tick
            return_payload = BikeReturnPayload(order.end_station)
            bike_return_evt = self._event_buffer.gen_atom_event(order.end_tick, BikeEventType.BikeReturn, payload=return_payload)

            self._event_buffer.insert_event(bike_return_evt)

    def _on_bike_return(self, evt: Event):
        payload: BikeReturnPayload = evt.payload
        target_station: Station = self._stations[payload.target_station]


        # TODO: what about more than capacity?
        target_station.inventory += 1

    def _on_action_recieved(self, evt: Event):
        action: Action  = None
        
        for action in evt.payload:
            station: Station = self._stations[action.station]

            executed_number = min(station.inventory, action.number)
            station.inventory -= executed_number

            payload = BikeTransferPaylod(action.station, action.to_station, action.number)
            transfer_evt = self._event_buffer.gen_atom_event(evt.tick + self._transfer_time, BikeEventType.BikeTransfered, payload)
            self._event_buffer.insert_event(transfer_evt)

    def _on_bike_recieved(self, evt: Event):
        payload: BikeTransferPaylod = evt.payload
        station: Station = self._stations[payload.to_station]

        # TODO: what about if out of capacity
        station.inventory += payload.number