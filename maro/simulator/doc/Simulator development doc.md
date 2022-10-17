# MARO simulator

MARO's simulator is the runtime environment of MARO. Its high-level structure is demonstrated in the following figure. We give a brief introdution of the entire structure in a top-down fashion.

![overview-overview](picture/overview-overview.svg#pic_center)




## `Env`

In maro, `Env` (Abbreviation for environment) is one of the most important components. It's the outermost layer of our runtime environment, but is has nothing to do with the specific scenario. Instead, it only provides a general interface that can be called to the scenario, so when using it, you need to create an exclusive `Env` instance for the specific scenario.

When creating it, it is necessary to specify `scenario`, `topology`, and other required parameters (such as `start_tick`, `durations`, and so on). For example:

```python
env = Env(scenario="cim", topology="global_trade.22p_l0.1", start_tick=0, durations=100)
```

The key object in `Env` is ++`step()`++, which is the interface that passes each step in the structure to push the next step of the environment. Here is a very simple example of how to interact with Env:

```python
metrics, decision_event, is_done = env.step(None)
```

When creating an `Env` instance, an instance of a special BusinessEngine (BE for short) is created internally through the `_init_business_engine()` method. The create `AbsBusinessEngine` instance (BE for short) interacts with the `Env` instance through `_simulate()` method. `_simulate()` method was developed through generators in Python language and is mainly used to drive the simulation scheduling. For the overall program, it is more like a channel linking BE and external decision makers, and the mutual information is exchanged and transferred.

## `Event Buffer`

`EventBuffer` implements the mutual communication between `Env` and BE. In MARO, there are two different types of events:

*   `AtomEvent`: Atomic event, i.e. a basic event without any additional functions or properties.
*   `CascadeEvent`: Events with child events. Sometimes (actually in most cases) there are dependencies between events, i.e., an event should be immediately executed right after another event is finished. This is when `CascadeEvent` is used: a `CascadeEvent` has a queue of sub-events. After the cascade event is executed, its sub-events will be immediately executed in order. A sub-event could also have sub-events, through which we can form a hierarchical structure.

There are two necessary parameters for any type of event:

*   `payload`: user-defined payload that contains actual contents of the event.
*   `event_type`: user-defined specific type of event, used to categorize the event.

You can defined your own event types of any data types. In MARO, there are two pre-defined event types:

*   `PENDING_DECISION`: used for decision events.
*   `TAKE_ACTION`: used for action events.

When BE needs an action from outside, it generates a decision event and throw it to `Env` (through `EventBuffer`). `Env` will pass the decision event with some process, and wait for agents to response. Once it receives the action from the agents, it will generate an action event that wraps the action, and pass the action event to BE (still, through EventBuffer). This is the general description of how `Env` and Be communicate with each other. We will back to this topic with many more details later.

## Business Engine

Before understanding BE, we need to understand `AbsBusinessEngine`, which is an abstract class and the base class for all scenarios. When developing a new scene, the developer should implement the new BE by inheriting this base class.

For a newly developed scenarios, all interfaces required by `AbsBusinessEngine` should be implemented, and then the exclusive scenarios logic can be personalized and implemented in any way you like. However, to make BE more efficient, MARO provides two sets of toolkits:

*   `EventBuffer`: In addition to connecting env and BE, this component can also manage the event logic of BE. Compared with processing all events in a loop, the efficiency of using BE management is much higher. Here is a simple example. When using `Eventbuffer` to manage the event flow, we only need to process a specific action on a specific time. The implication is that we only need to wait for a specific time to arrive and process it without repeatedly traversing the check time.(In the program, the variable represented by this particular time is the `tick`)
*   `Frame`: `Frame` is a low-level data structures implemented using Cython. It is much more faster to read and write for structual data.

Although the event logic of be is organized by event buffer, we should understand what is `event handler` before using it.

#### Event Handler

Event handler is a built-in handler for events in the `Eventbuffer`. Its role is to bind events to handler functions, and select different handler functions by judging different types of events (type is initially defined by the user). Now give an example to show you how to deal with it:

    {
        evet_type_A: Function_A,
        evet_type_B: Function_B,
    }

This is a dictionary-type structure containing different functions for different types of events. When the event type is type A, `EventBuffer` will automatically bind it to function A and use that function to handle the event. With the support of `EventBuffer`, BE only needs to complete some other things:

*   Define event types.
*   Define processing logic (function) for each type of event (including action event).
*   Register the mapping of event type and processing logic (function) in the event buffer's event handler.
*   Put initial events into event buffer.

## Event Loop

After understanding `Eventbuffer` and `AbsBusinessEngine`, let's talk about the life cycle of events in MARO.

As mentioned above, `Env` will first pass the decision event to the outside, and then the action will be sent back from the outside, and the action will be packaged as an event into the `Env`. With the change of the system `tick`, the event buffer will control the processing order, that is, the current atom event will be processed under a specific tick, and the decision event will be returned.

For these events, we constructed the `EventLinkedList` data structure to store them. Currently, some methods in this structure are currently defined:

*   `append_tail` (`append`): This method can be used when a new event needs to be added, which appends the event to the tail.
*   `append_head` : Insert an event to the head, will be the first one to pop.
*   `_extract_sub_events` : to extract immediate events to the head.
*   `clear_finished_and_get_front`: Empty completed events and get unfinished first.

![overview-event](picture/overview-event.svg#pic_center)

For the overall program, each event has a start and end state. As shown in the figure, when the system tick starts, the handler will automatically process the `AtomEvent` under the current tick and check whether there is a cascadeevent. When the processing is completed, the end event will be appended to a specific time.
