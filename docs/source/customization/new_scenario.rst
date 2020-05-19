New Scenario
=============


Prerequisites
~~~~~~~~~~~~~

1. Currently MARO only support search scenarios under its own folder "maro/simulator/scenarios",
2. Default environment simulator only search files that named as "business_engine.py" under each scenario folder. 


We have a built-in scenario "sample" used to show how to make a scenario with mock business logic, you can refer it for more details.
Following steps will use this sample scenario as example to explain what you need to create a new scenario.

Steps to make a new scenario
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Step 1 - Scenario folder
------------------------

Create a new folder under "maro/simulator/scenarios", name it as what you want, such as "sample". 
And create a folder named as "topologies" to hold your configurations, a sub-folder is a configration name, the content format is depend on your requirement,
MARO will not read it, just combine the configuration folder path, and pass it to you, so you can use any format you like.


Step 2 - Data mode definition
-----------------------------

Usually we should define our data model first, MARO provided high level API (inspired by django) to make it easier to define data model.


MARO frame comes with 2 node type, STATIC and DYNAMIC, you can save your model in these node, one data model one node type, so you can have only 2 data model types for now.


You can define any attributes with XXXAttribute wrapper for different data type, such as following example, the data mode class "SampleStaticModel" use STATIC node to hold attributes,
and it contains 2 attributes, "a" is a integer attribute, b is a float array that length is 2.


Sometimes you may want to handle special attributes' values changes to fit your logic, MARO support an implicit way to for this, if you 
have a function named match "_on_<attribute_name>_changed", then it will be invoked every time the value of attribute "a" changed. 


.. code-block:: python

    @frame_node(STATIC_NODE)
    class SampleStaticModel(EntityBase):
        # the node type for attribute used to access value
        a = IntAttribute() # add an int attribute "a" with 1 slot (default)
        b = FloatAttribute(slot_num=2)

        # NOTE: since your implementation of nodes is an array, so we need the index instead of id (or other identifier)
        # but you can create your own mapping out-side
        def __init__(self, frame: Frame, index: int):
            super().__init__(frame, index)

        # NOTE: value changed callbacks are binded automatically if name match "_on_<attribute name>_changed"
        def _on_a_changed(self, slot_index: int, new_val):
            print(f"value of a changed to {new_val} at slot {slot_index}")


Step 3 - Implement business logic


As described above, your have to have a file named as "business_engine.py" under your scenario folder, and you should inherit from
AbsBusinessEngine and implement all the abstract methods and properties.


There are some most important points that your have to do in your business engine.

1. Initialize Frame with your data model definition using FrameBuilder, and create a new instance of SnapshotList, this is the only way 
out-side of environment can access your data models.


This is the code snippet from sample scenario (maro/simulator/scenarios/sample/business_engine.py).


.. code-block:: python

    # use helper to build the frame (optional)
    self._frame = FrameBuilder.new() \
        .add_model(SampleStaticModel, STATIC_NODE_NUM) \
        .add_model(SampleDynamicModel, DYNAMIC_NODE_NUM) \
        .build()
            
    #build_frame(SampleStaticModel, STATIC_NODE_NUM, SampleDynamicModel, DYNAMIC_NODE_NUM)

    # then we can build the snapshot list
    # NOTE: the frame_resolution used to control the frequency to take snapshot
    # total_frames is a helper function to calculate total frames in snapshot list
    self._snapshot_list = SnapshotList(self._frame, total_frames(self._start_tick, self._max_tick, self._frame_resolution))


2. Since MARO simulator is event-driven, you need to define you events and register them with callbacks.
MARO has a built-in event "DECISION_EVENT" that used to send actions from agent to simulator, so you should at least handle this event.


.. code-block:: python
    
    # our events 
    class SampleEventType(IntEnum):
        Event1 = 10
        Event2 = 11
        Event3 = 12

    # register your event with callback handler
    # when there is any event that match the type at current tick, callback functions will be invoked
    self._event_buffer.register_event_handler(SampleEventType.Event1, self._on_event1)
    self._event_buffer.register_event_handler(SampleEventType.Event2, self._on_event2)

    # this is the pre-defined event, used to handle actions from agents
    self._event_buffer.register_event_handler(DECISION_EVENT, self._on_action_recieved)


NOTES:

1. Your business engine should decide when to stop the simulation, you can return True to stop in functions "step" and "post_step".
Step would be called several times for each tick, post_step will be called after all the events have been processed.

2. Default simulator (Env class) will only take snapshot from the Frame provided by your business engine, but you can take snapshot for any tick as you like with "snapsots.take_snapshot".


Step 4 -- Executing


After above 3 steps, you have a scenario now, then you can write a start script to interact with scenario name, configuration name, and max tick.
Then use for loop to step our simulator until the end.

.. code-block:: python

    env = Env(scenario="sample", topology="sample",  max_tick=MAX_TICK)

    reward = None # reward from simulator
    decision_event: DecisionEvent = None # decision event from simulator to ask an action
    is_done: bool = False # if simulator reach the end

    action = Action(0, 1111) # our dummy action

    # reset our env first
    env.reset()

    # NOTE: we must pass None at first step at each episode
    reward, decision_event, is_done = env.step(None)

    while not is_done:
        reward, decision_event, is_done = env.step(action)


NOTE:

Please make that you have set the PYTHONPATH to the root folder of MARO, and build the extension with build_maro.sh script.
