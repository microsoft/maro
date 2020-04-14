from enum import IntEnum

from .common import DecisionEvent
from .model import SampleStaticModel, SampleDynamicModel
from maro.simulator.utils.random import random
from maro.simulator.scenarios import AbsBusinessEngine
from maro.simulator.scenarios.modelbase import build_frame
from maro.simulator.frame import Frame, SnapshotList
from maro.simulator.utils.common import total_frames, tick_to_frame_index
from maro.simulator.event_buffer import EventBuffer, DECISION_EVENT, Event

# NOTE: if you want your random sequence can be reproduce, you should use random function provided by simulator,
# then your can use maro.simulator.utils.random.random.seed function to set seed for all the randon sequence

# create a new random sequence, named as sample
sample_random = random["sample"]

STATIC_NODE_NUM = 5
DYNAMIC_NODE_NUM = 6

## STEP 2: define your event type
# NOTE: we already have a pre-defined event DECISION_EVENT which value is 0, make sure not override this
class SampleEventType(IntEnum):
    Event1 = 10
    Event2 = 11
    Event3 = 12

## STEP 3: define your business logic
class SampleBusinessEngine(AbsBusinessEngine):
    def __init__(self, event_buffer: EventBuffer, config_path: str, start_tick: int, max_tick: int, frame_resolution: int):
        super().__init__(event_buffer, config_path, start_tick, max_tick, frame_resolution)
        
        self._frame: Frame = None
        self._snapshot_list: SnapshotList = None
        self._static_nodes: list = []
        self._dynamic_nodes: list = []
        
        # read and parse your config from config_path (under topologies/xxx/config.yml)
        # we ignore it here

        # register your events
        self._reg_events()
        # setup frame and nodes
        self._init_frame()

    @property
    def frame(self):
        return self._frame

    @property
    def snapshots(self):
        return self._snapshot_list

    @property
    def configs(self):
        # any configuratins you need to exposed
        return {}

    def rewards(self, actions: list):
        # calculate you rewards
        return []

    def step(self, tick: int) -> bool:
        # this function is called every tick, tick if current tick
        # usually we load/generate our business data and push into EventBuffer to use related handlers process it

        ## we just a random value here
        val = sample_random.uniform(-10, 10)

        # usually these data should be processed at current tick, so we specified the event tick as current tick,
        # but you can delay it to any tick as wish (must less than max_tick)

        # NOTE: atom is normal event, they will not trigger any decision
        evt = self._event_buffer.gen_atom_event(tick, SampleEventType.Event1, payload=val)

        # insert into EventBuffer, so that the hanlder will handle it
        self._event_buffer.insert_event(evt)

        ### then it is the most important part, we need to decide if we should ask agent for an action
        # you can do this any where by insert a cascade event with related payload

        # for sample, we just ask an action every 5 ticks
        if (tick + 1) % 5 == 0:
            # though it named as DecisionEvent, but is act as a payload here
            payload = DecisionEvent(0, self._action_scope)

            decition_evt = self._event_buffer.gen_cascade_event(tick, DECISION_EVENT, payload)

            self._event_buffer.insert_event(decition_evt)

        # the return value means if we should stop at current tick, usually we stop at the last tick as following
        return tick + 1 == self._max_tick

    def post_step(self, tick: int):
        # this function is called after events of current tick processed 
        
        # usually we should check if we should take a snapshot that match the frame_resolution, but it is not required
        if (tick + 1) % self._frame_resolution == 0:
            snapshot_index = tick_to_frame_index(self._start_tick, tick, self._frame_resolution)

            self._snapshot_list.insert_snapshot(self._frame, snapshot_index)

    def reset(self):
        # any reset logic here after each episode
        # usually clear the snapshot and initialize the node

        # NOTE: following 2 reset function will make all the attributes to be 0, need to initialize your value again
        self._frame.reset()
        self._snapshot_list.reset()

        # initial you nodes
        self._init_nodes()

    def get_node_name_mapping(self):
        # your mapping for out-side using
        return {}

    def get_agent_idx_list(self):
        # return index list of your agents, such as Ports for ECR
        return []

    def _action_scope(self, node_index: int):
        node: SampleStaticModel = self._static_nodes[node_index]

        return node.a

    def _init_nodes(self):
        # initialize the value of your nodes, we only initialize first static node here
        node: SampleStaticModel = self._static_nodes[0]

        node.a = 123
        node.b[1] = 1

    def _init_frame(self):
        # initialize your frame with data model
        # say we have 5 static node, 6 dynamic nodes, usually this may come from config file

        # use helper to build the frame (optional)
        self._frame = build_frame(SampleStaticModel, STATIC_NODE_NUM, SampleDynamicModel, DYNAMIC_NODE_NUM)

        # then we can build the snapshot list
        # NOTE: the frame_resolution used to control the frequency to take snapshot
        # total_frames is a helper function to calculate total frames in snapshot list
        self._snapshot_list = SnapshotList(self._frame, total_frames(self._start_tick, self._max_tick, self._frame_resolution))

        # NOTE: currently we have assign the index manually, so you can create you own mapping here
        for i in range(STATIC_NODE_NUM):
            self._static_nodes.append(SampleStaticModel(self._frame, i))

        for i in range(DYNAMIC_NODE_NUM):
            self._dynamic_nodes.append(SampleDynamicModel(self._frame, i))
        
        self._init_nodes()

    def _reg_events(self):
        # register your event with callback handler
        # when there is any event that match the type at current tick, callback functions will be invoked
        self._event_buffer.register_event_handler(SampleEventType.Event1, self._on_event1)
        self._event_buffer.register_event_handler(SampleEventType.Event2, self._on_event2)

        # this is the pre-defined event, used to handle actions from agents
        self._event_buffer.register_event_handler(DECISION_EVENT, self._on_action_recieved)

    def _on_event1(self, evt: Event):
        # payload used to hold information to process current event, can be any object
        payload = evt.payload

        print("event 1", payload)

        # you can insert a new event inside a callback function to chain your all logic pieces
        evt = self._event_buffer.gen_atom_event(evt.tick + 1, SampleEventType.Event2, (1, 2, 3))

        self._event_buffer.insert_event(evt)


    def _on_event2(self, evt: Event):
        print("event 2", evt.payload)


    def _on_action_recieved(self, evt: Event):  
        print("recieved an action", evt.payload)

        # we hard code the proccessing, so we can see the changes of attribute 'b' from outside
        node: SampleStaticModel = self._static_nodes[0]
        node.b[1] = evt.tick