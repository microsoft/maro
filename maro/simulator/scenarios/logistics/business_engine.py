import os
from enum import IntEnum
from typing import List, Dict

import numpy as np
from yaml import safe_load

from maro.simulator.frame import Frame, SnapshotList, FrameAttributeType, FrameNodeType
from maro.simulator.event_buffer import Event, EventBuffer, DECISION_EVENT
from maro.simulator.scenarios import AbsBusinessEngine
from maro.simulator.scenarios.entity_base import FrameBuilder
from .warehouse import Warehouse
from .common import Action, Demand, DecisionEvent


STATIC_NODE = FrameNodeType.STATIC
WAREHOUSE_NUMBER = 1


class LogisticsType(IntEnum):
    DemandRequirement = 1  # 


class LogisticsBusinessEngine(AbsBusinessEngine):
    def __init__(self, event_buffer: EventBuffer, config_path: str, start_tick: int, max_tick: int, frame_resolution: int):
        super().__init__(event_buffer, config_path, start_tick, max_tick, frame_resolution)

        # these items will update from configuration file
        self._init_stock = 0
        self._max_order = 10
        self._max_capacity = 100
        self._min_demand = 0
        self._max_demand = 10
        self._min_demand_noise = -1
        self._max_demand_noise = 1
        self._unit_cost = 1
        self._unit_price = 2
        self._storage_cost = 0.5

        self._profit = 0  # used for calculating reward
        self._frame: Frame = None
        self._snapshot_list: SnapshotList = None
        self._warehouses = []
        self._conf = {}

        self._init_config()
        self._register_events()
        self._init_frame()

    ###### minimum functions inherit from AbsBusinessEngine, other functions can be override depend on requirements ########

    def step(self, tick: int) -> bool:
        demand_value = self.demand(tick) # generate a demand by tick

        # generate and insert an event
        # though the event's payload can be any object, here we just use the demand number as payload
        demand_evt = self._event_buffer.gen_atom_event(tick, LogisticsType.DemandRequirement, payload=Demand(demand_value))
        self._event_buffer.insert_event(demand_evt)

        # NOTE: we can stop the simulator at step and post_step
        # if we reach the end?
        # also we can return false, and move this line into post_step to simple the checking logic
        return (tick + 1) == self._max_tick

    def post_step(self, tick: int) -> bool:
        # this function will be invoked after all the events of current tick being procesed
        # do anything that need to process here
        
        # since we use event to process the demands, the new stock value cannot be checked in step,
        # so we check it here

        # shall we stop after processed all the event?
        is_done = not all([0 <= w.stock <= w.max_capacity for w in self._warehouses])

        # let's take the snapshot, as env will not take snapshot at end (only before take action),
        # then we can see the value changes
        self._snapshot_list.insert_snapshot(self._frame, tick)

        return is_done

    def rewards(self, actions):
        # ignore actions, just return recorded profit
        return [self._profit] # our reward must be a list for now

    @property
    def frame(self) -> Frame:
        return self._frame

    @property
    def snapshots(self) -> SnapshotList:
        return self._snapshot_list

    def get_node_name_mapping(self) -> dict:
        """
        Get node name mappings related with this environment

        Returns:
            Node index to name mapping dictionary
            {
                "static": {index: name}
            }
        """
        return dict(static={w.index: w.name for w in self._warehouses})

    def get_agent_idx_list(self) -> List[int]:
        """Get warehouse index list related with this environment
        Returns:
            List[int]: list of port index
        """
        return list(range(len(self._warehouses)))

    def reset(self):
        # reset frame will clear the value of stock to 0, so we need to reset warehouse here
        self._frame.reset()
        self._snapshot_list.reset()

        for warehouse in self._warehouses:
            warehouse.reset()

    ###### private functions ######

    def demand(self, tick: int):
        weekday_trend = (np.cos((tick % 7) * 2 * np.pi / 7) + 1) / 2
        scaled = (self._max_demand - self._min_demand) * weekday_trend + self._min_demand
        noise = np.random.randint(self._min_demand_noise, self._max_demand_noise + 1)
        return np.maximum(0, int(scaled) + noise)

    def _register_events(self):
        self._event_buffer.register_event_handler(LogisticsType.DemandRequirement, self._on_demand_requirement)
        self._event_buffer.register_event_handler(DECISION_EVENT, self._on_action_recieved)

    def _init_frame(self):
        # only build the frame with static node (Warehouse)
        self._frame = (FrameBuilder
                       .new()
                       .add_model(Warehouse, WAREHOUSE_NUMBER)
                       .build())  # how many agents = 1

        # create our snapshot_list, and specified how many snapshots we can take
        self._snapshot_list = SnapshotList(self._frame, self._max_tick)

        # create warehouse instance to acess
        self._warehouses = [
            Warehouse(
                index=i,
                initial_stock=self._init_stock,
                max_capacity=self._max_capacity,
                frame=self._frame
            ) for i in range(WAREHOUSE_NUMBER)
        ]

        # this would read recorded data for playback

    def _init_config(self):
        with open(os.path.join(self._config_path, "config.yml")) as fp:
            self._conf = safe_load(fp)

        self._max_order = self._conf["max_order"]
        self._max_capacity = self._conf["max_capacity"]
        self._init_stock = self._conf["init_stock"]
        self._min_demand = self._conf["min_demand"]
        self._max_demand = self._conf["max_demand"]
        self._min_demand_noise = self._conf["min_demand_noise"]
        self._max_demand_noise = self._conf["max_demand_noise"]
        self._unit_cost = self._conf["unit_cost"]
        self._unit_price = self._conf["unit_price"]
        self._storage_cost = self._conf["storage_cost"]

    def _action_scope(self, index, event_type=None):
        return {w.index: self._max_order for w in self._warehouses}

    def _on_demand_requirement(self, evt: Event):
        demand: Demand = evt.payload

        for warehouse in self._warehouses:
            warehouse.fulfill_demand(evt.tick, demand.demand)
            if 0 <= warehouse.stock <= warehouse.max_capacity:
                # create payload for decision event
                decision_event = DecisionEvent(evt.tick, warehouse.index, self.snapshots, self._action_scope)
                event = self._event_buffer.gen_cascade_event(evt.tick, DECISION_EVENT, payload=decision_event)
                self._event_buffer.insert_event(event)

    def _on_action_recieved(self, evt: Event):
        # here we recieved actions (Env will wrapper actions into a list, even only one action)
        actions: Action = evt.payload

        # restock warehouses and calculate profit
        profit = 0
        for action in actions:
            warehouse = self._warehouses[action.warehouse_idx]
            warehouse.supply_stock(action.restock)
            profit += warehouse.fulfilled * self._unit_price
            profit -= warehouse.unfulfilled * self._unit_price
            profit -= warehouse.stock * self._storage_cost 
            profit -= action.restock * self._unit_cost

        self._profit = profit
