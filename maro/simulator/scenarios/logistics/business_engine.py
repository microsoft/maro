import os
from enum import IntEnum
from typing import List, Dict
from maro.simulator.frame import Frame, SnapshotList, FrameAttributeType, FrameNodeType
from maro.simulator.event_buffer import Event, EventBuffer, DECISION_EVENT
from maro.simulator.scenarios import AbsBusinessEngine
from maro.simulator.scenarios.entity_base import build_frame
from .warehouse import Warehouse
from .common import Action, Demand
from yaml import safe_load

STATIC_NODE = FrameNodeType.STATIC

WAREHOUSE_NUMBER = 1

class LogisticsType(IntEnum):
    DemandRequirement = 1  # 

class LogisticsBusinessEngine(AbsBusinessEngine):
    def __init__(self, event_buffer: EventBuffer, config_path: str, start_tick: int, max_tick: int, frame_resolution: int):
        super().__init__(event_buffer, config_path, start_tick, max_tick, frame_resolution)

        # these 4 items will update from configuration file
        self._max_order = 0
        self._max_capacity = 0
        self._demand_step = 1
        self._init_stock = 0

        self._frame: Frame = None
        self._snapshot_list: SnapshotList = None
        self._warehouses = []
        self._conf = {}

        self._cur_demand = 0 # used to hold current demand for reward function

        self._init_config()
        self._register_events()
        self._init_frame()

    ###### minimum functions inherit from AbsBusinessEngine, other functions can be override depend on requirements ########

    def step(self, tick: int) -> bool:
        demand_number = self.demand(tick) # generate a demand by tick
        self._cur_demand = demand_number

        # generate and insert an event
        # though the event's payload can be any object, here we just use the demand number as payload
        demand_evt = self._event_buffer.gen_atom_event(tick, LogisticsType.DemandRequirement, payload=Demand(demand_number))

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
        stock = self._warehouses[0].stock
        is_done = stock < 0 or stock > self._max_capacity

        # let's take the snapshot, as env will not take snapshot at end (only before take action),
        # then we can see the value changes
        self._snapshot_list.insert_snapshot(self._frame, tick)

        return is_done

    def rewards(self, actions):
        # ignore actions, just return current demand as rewards
        return [self._cur_demand] # our reward must be a list for now

    @property
    def frame(self) -> Frame:
        return self._frame

    @property
    def snapshots(self) -> SnapshotList:
        return self._snapshot_list

    def get_agent_idx_list(self) -> List[int]:
        """Get port index list related with this environment
        Returns:
            List[int]: list of port index
        """
        return [i for i in range(len(self._warehouses))]

    def reset(self):
        # reset frame will clear the value of stock to 0, so we need to reset warehouse here
        self._frame.reset()
        self._snapshot_list.reset()

        for warehouse in self._warehouses:
            warehouse.reset()

    ###### private functions ######

    def demand(self, tick: int):
        return tick % self._demand_step # TODO: catch the exception

    def _register_events(self):
        self._event_buffer.register_event_handler(LogisticsType.DemandRequirement, self._on_demand_requirement)
        self._event_buffer.register_event_handler(DECISION_EVENT, self._on_action_recieved)

    def _init_frame(self):
        # only build the frame with static node (Warehouse)
        self._frame = build_frame(Warehouse, WAREHOUSE_NUMBER) # # how many agents = 1

        # create our snapshot_list, and specified how many snapshots we can take
        self._snapshot_list = SnapshotList(self._frame, self._max_tick)

        # create warehouse instance to acess
        for i in range(WAREHOUSE_NUMBER):
            self._warehouses.append(Warehouse(i, self._init_stock, self._max_capacity, self._frame))

        # this would read recorded data for playback

    def _init_config(self):
        with open(os.path.join(self._config_path, "config.yml")) as fp:
            self._conf = safe_load(fp)

            self._max_order = self._conf["max_order"]
            self._max_capacity = self._conf["max_capacity"]
            self._demand_step = self._conf["demand_step"]
            self._init_stock = self._conf["init_stock"]

    def _on_demand_requirement(self, evt: Event):
        demand: Demand = evt.payload

        self._warehouses[0].fulfill_demand(demand.demand)

        stock = self._warehouses[0].stock

        if stock >=0 and stock<= self._max_capacity:
            # TODO: we need an action each tick?
            # for this demo, we do not need any payload for decision event
            decision_evt = self._event_buffer.gen_cascade_event(evt.tick, DECISION_EVENT, None)

            self._event_buffer.insert_event(decision_evt)

    def _on_action_recieved(self, evt: Event):
        # here we recieved actions (Env will wrapper actions into a list, even only one action)
        action: Action = evt.payload[0]

        self._warehouses[0].supply_stock(action.restock)