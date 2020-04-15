from enum import IntEnum
from maro.simulator.frame import Frame, SnapshotList, FrameAttributeType, FrameNodeType
from maro.simulator.event_buffer import Event, EventBuffer, DECISION_EVENT
from maro.simulator.scenarios import AbsBusinessEngine

STATIC_NODE = FrameNodeType.STATIC

class LogisticsType(IntEnum):
    DemandRequirement = 1  # 

class LogisticsBusinessEngine(AbsBusinessEngine):
    def __init__(self, event_buffer: EventBuffer, config_path: str, start_tick: int, max_tick: int, frame_resolution: int):
        super().__init__(event_buffer, config_path, start_tick, max_tick, frame_resolution)

        self._frame: Frame = None
        self._snapshot_list: SnapshotList = None
        self._max_capacity = 15
        self._max_order = 5
        self._cur_demand = 0 # used to hold current demand for reward function

        self._register_events()
        self._init_frame()

    ###### minimum functions inherit from AbsBusinessEngine, other functions can be override depend on requirements ########

    def step(self, tick: int) -> bool:
        demand_number = self.demand(tick)
        self._cur_demand = demand_number

        # generate and insert an event
        # though the event's payload can be any object, here we just use the demand number as payload
        demand_evt = self._event_buffer.gen_atom_event(tick, LogisticsType.DemandRequirement, payload=demand_number)

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
        is_done = self.stock < 0 or self.stock > self._max_capacity

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

    def reset(self):
        # reset frame will clear the value of stock to 0, so we do not reset stock here
        self._frame.reset()
        self._snapshot_list.reset()

    ###### private functions ######

    def _register_events(self):
        self._event_buffer.register_event_handler(LogisticsType.DemandRequirement, self._on_demand_generated)
        self._event_buffer.register_event_handler(DECISION_EVENT, self._on_action_recieved)

    def _init_frame(self):
        # here we use raw api to build a frame for demo, refer to sample scenario to see how to use helpers for complex logic
        # we only have 1 static node (pre-defined in Frame) to hold order
        self._frame = Frame(1, 0)

        # register an attribute with one slot as int type
        self._frame.register_attribute("stock", FrameAttributeType.INT,1)
        self._frame.setup() # setup will allocate memory based on registered attributes

        # create our snapshot_list, and specified how many snapshots we can take
        self._snapshot_list = SnapshotList(self._frame, self._max_tick)

    # wrapper to access Frame with raw api
    @property
    def stock(self):
        # we assume our stock same as static node
        return self._frame.get_attribute(STATIC_NODE, 0, "stock", 0)

    @stock.setter
    def stock(self, value):
        self._frame.set_attribute(STATIC_NODE, 0, "stock", 0, value)

    def demand(self, tick: int):
        return tick % 3

    def _on_demand_generated(self, evt: Event):
        demand_number: int = evt.payload

        self.stock -= demand_number

        if self.stock >=0 and self.stock<= self._max_capacity:
            # TODO: we need an action each tick?
            # for this demo, we do not need any payload for decision event
            decision_evt = self._event_buffer.gen_cascade_event(evt.tick, DECISION_EVENT, None)

            self._event_buffer.insert_event(decision_evt)

    def _on_action_recieved(self, evt: Event):
        # here we recieved actions (Env will wrapper actions into a list, even only one action)
        # we assuming that agent will only pass a number as action here
        action_number: int = evt.payload[0]

        self.stock += action_number