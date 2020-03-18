
import os
import csv

from typing import Dict, List
from maro.simulator.graph import Graph, SnapshotList
from yaml import safe_load
from maro.simulator.event_buffer import EventBuffer, DECISION_EVENT, Event
from maro.simulator.scenarios import AbsBusinessEngine
from .station import Station
from .graph_builder import build
from .data_reader import BikeDataReader
from enum import IntEnum

class BikeEventType(IntEnum):
    Order = 10
    BikeReturn = 11

class BikeBusinessEngine(AbsBusinessEngine):
    def __init__(self, event_buffer: EventBuffer, config_path: str, max_tick: int):
        super().__init__(event_buffer, config_path)

        self._max_tick = max_tick
        self._stations = []
        self._station_map = {}

        config_path = os.path.join(config_path, "config.yml")

        self._conf = None

        with open(config_path) as fp:
            self._conf = safe_load(fp)

        self._init_graph()
        self._init_data_reader()
        
        self._snapshots = SnapshotList(self._graph, max_tick)

    @property
    def graph(self) -> Graph:
        """Graph: Graph of current business engine
        """
        return self._graph

    @property
    def snapshots(self) -> SnapshotList:
        """SnapshotList: Snapshot list of current graph"""
        return self._snapshots

    def step(self, tick: int):
        """Used to process events at specified tick, usually this is called by Env at each tick

        Args:
            tick (int): tick to process
        """
        orders = self._data_reader.get_orders(tick)

        for order in orders:
            print(order)
            # self._event_buffer.gen_atom_event(tick, )

    @property
    def configs(self) -> dict:
        """object: Configurations of this business engine"""
        return {}

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

    def action_scope(self, port_idx: int, vessel_idx: int) -> object:
        """Get the action scope of specified agent

        Args:
            port_idx (int): Port index of specified agent
            vessel_idx(int): Vessel index to take the action

        Returns:
            object: action scope object that may different for each scenario
        """
        pass

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
        return []

    def post_step(self, tick):
        """Post-process at specified tick

        Args:
            tick (int): tick to process

        """
        pass

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

            station = Station(i, int(r[0]), int(r[2]), self._graph)

            self._stations.append(station)
            self._station_map[int(r[0])] = i

    def _init_data_reader(self):
        self._data_reader = BikeDataReader(self._conf["data_file"], self._conf["start_datetime"], self._max_tick, self._station_map)

    
    def _reg_event(self):
        self._event_buffer.register_event_handler(BikeEventType.Order, self._on_order_gen)
        self._event_buffer.register_event_handler(BikeEventType.BikeReturn, self._on_bike_return)


    def _on_order_gen(self, evt: Event):
        pass

    def _on_bike_return(self, evt: Event):
        pass