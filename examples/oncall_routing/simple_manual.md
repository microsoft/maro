
# On-Call Routing Scenario

## Features

### Decision event at a glance

Each time the simulator needs agents to take actions, it will suspend simulation and throw a decision event. A decision event contains following attributes.

```python
oncall_orders: List[Order]
route_plan_dict: Dict[str, List[Route]]
route_meta_info_dict: Dict[str, dict]
estimated_duration_predictor: EstimatedDurationPredictor
```

### `oncall_orders`

On-call orders to be processed. For an on-call order, we need to pay attention to the following attributes / methods of the `Order` class:

- `id (str)`: ID of the order.
- `coordinate (Coordinate)`: The coordinate of the order. `Coordinate` class is a namedtuple which contains a latitude (`lat`) and a longitude (`lng`), which is defined in `maro/simulator/scenarios/oncall_routing/coordinate.py`.
- `open_time (int)`: Open / ready time of the order.
- `close_time (int)`: Close time of the order.
- `creation_time (int)`: The time that this oncall order is created.
- `get_status(self, tick: int, transition_config) -> OrderStatus`: Get the status of the order at the current tick. The `OrderStatus` is an Enum class defined in `maro/simulator/scenarios/oncall_routing/order.py`.

*The `Order` class is defined in `maro/simulator/scenarios/oncall_routing/order.py`.*

### `route_plan_dict`

A dict that contains remaining planned orders for each route, using route name as keys. For example, we could get route A's remaining planned orders by calling `decision_event.route_plan_dict["A"]`. The attributes of an `Order` object can be found in the last section. For each route, the planned orders would be processed in the order they are in the plan list.

### `route_meta_info_dict`

A dict that contains a set of meta-info of each route, using route name as keys. The meta-info contains:

- `carrier_idx`: The index of the corresponding carrier of this route.
- `estimated_duration_to_the_next_stop`: The estimated travel time from the carrier's current location to its next destination (following the route plan). If the carrier has finished all the planned orders, this value would be `None`.
- `estimated_next_departure_tick`: The estimated departure time for the corresponding carrier of this route to process the orders in current stop. If the carrier is not in the stop at this moment, this value would be `None`.

### `estimated_duration_predictor`

An `EstimatedDurationPredictor` object used to predict the travel time between two positions, by calling:

```python
EstimatedDurationPredictor.predict(
    tick: int,  # The tick to start travelling.
    source_coordinate: Coordinate,
    target_coordinate: Coordinate,
)
```

Agents could use this predictor to predict the travel time between stops.

*Currently, the returned predictor is time-independent.*

## More features from the Env

### `carriers_in_stop`

`carriers_in_stop` is not listed in the `decision_event`, but it can be obtained through `(env.snapshot_list["carriers"][env.tick::"in_stop"] == 1).tolist()`. It is a list of `bool` values, indicating whether carriers are in a stop or not. If a carrier is in a stop, it is allowed to insert oncall orders before the carrier's next stop. Otherwise, it means the carrier is already on its way to the next stop, so it it not allowed to insert oncall orders before the carrier's current destination.

## Actions

The actions given to the simulator are formulated as a list of `Action` objects.
There are 2 types of valid `Action`: `AllocateAction` and `PostponeAction` respectively.

The `PostponeAction` is used to indicate that we'll not allocate any route for this on-call order now, and the on-call request would be given again later. The `PostponeAction` has only 1 attribute:

- `order_id (str)`: The ID of the on-call order to be processed.

The `AllocateAction` is used when you want to insert an on-call order to a specific location of a specific route. An `Action` object has four attributes:

- `order_id (str)`: The ID of the oncall order to be processed.
- `route_name (str)`: The name of the route where the oncall order is to be inserted.
- `insert_index (int)`: Insert the oncall order BEFORE the `insert_index` stop of the current plan. For example, `insert_index = 5` means we want to insert the oncall order before the 5th stop of the current plan.
- `in_segment_order (int)`: This is used when we want to insert multiple oncall orders into the same position of the current plan. In this case, we need to explicitly tell the simulator the relative order of these oncall orders through `in_segment_order`. The smaller, the earlier. For example, if we have two actions: `a1 = Action(order_id="1", route_name="1", insert_index=5, in_segment_order=0)` and `a2 = Action(order_id="2", route_name="1", insert_index=5, in_segment_order=1)`, it means we want to insert both `a1` and `a2` before the 5th stop of route `1`, but `a1` is placed before `a2`.

Each time a decision event occurs, we need to return a list of `Action` objects to the simulator. The oncall orders that are not included in the actions will be ignored and would be thrown out in the next decision event again.
