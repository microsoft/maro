from enum import IntEnum
from maro.simulator.frame import Frame, SnapshotList, FrameAttributeType, FrameNodeType
from maro.simulator.event_buffer import Event, EventBuffer, DECISION_EVENT
from maro.simulator.scenarios import AbsBusinessEngine

STATIC_NODE = FrameNodeType.STATIC

class LogisticsType(IntEnum):
    """Events we need to handled to process trip logic"""
    DemandRequirement = 10  # 

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

    ###### functions inherit from AbsBusinessEngine ########

    def step(self, tick: int) -> bool:
        demand_number = self.demand(tick)
        self._cur_demand = demand_number

        # generate and insert an event
        # though the event's payload can be any object, here we just use the demand number as payload
        demand_evt = self._event_buffer.gen_atom_event(tick, LogisticsType.DemandRequirement, payload=demand_number)

        self._event_buffer.insert_event(demand_evt)

        # TODO: we need an action each tick?
        # for this demo, we do not need any payload for decision event
        decision_evt = self._event_buffer.gen_cascade_event(tick, DECISION_EVENT, None)

        self._event_buffer.insert_event(decision_evt)

        # if we reach the end
        return self.stock < 0 or self.stock > self._max_capacity or (tick + 1) == self._max_tick

    def post_step(self, tick: int):
        # this function will be invoked after all the events process of current tick
        # do anything that need to process here at the end of current tick
        pass

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

    def _on_action_recieved(self, evt: Event):
        # here we recieved actions (Env will wrapper actions into a list, even only one action)
        # we assume agent will only pass a number as action here
        action_number: int = evt.payload[0]

        self.stock += action_number