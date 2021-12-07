# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from collections import defaultdict, deque
from itertools import count
from typing import Deque, Dict, List, Optional

from yaml import safe_load

from maro.backends.frame import FrameBase, SnapshotList
from maro.data_lib.oncall_routing import get_data_loader, get_oncall_generator
from maro.event_buffer import AtomEvent, CascadeEvent, EventBuffer, MaroEvents
from maro.simulator.scenarios import AbsBusinessEngine
from maro.simulator.scenarios.helpers import DocableDict
from maro.simulator.utils import random
from maro.utils import convert_dottable

from .carrier import Carrier
from .common import (
    Action, CarrierArrivalPayload, CarrierDeparturePayload, Events, OncallReceivePayload, OncallRoutingPayload,
    OrderProcessingPayload, PlanElement
)
from .coordinate import Coordinate
from .duration_time_predictor import ActualDurationSampler, EstimatedDurationPredictor
from .frame_builder import gen_oncall_routing_frame
from .order import GLOBAL_ORDER_COUNTER, Order, OrderStatus
from .route import Route
from .utils import GLOBAL_RAND_KEY

metrics_desc = """
Oncall routing metrics used provide statistics information until now.
It contains following keys:

total_oncall_num(int): The total number of oncall orders.
total_order_num(int): The total number of orders.
total_failures_due_to_early_arrival(int):
total_order_in_advance(int):
total_order_delayed(int): The total delayed order quantity of all carriers.
total_order_delay_time(int): The total order delay time of all carriers.
total_order_terminated(int): The total number of order that are terminated during the simulation.
total_order_completed(int):
"""


class OncallRoutingBusinessEngine(AbsBusinessEngine):
    def __init__(
        self,
        event_buffer: EventBuffer,
        topology: Optional[str],
        start_tick: int,
        max_tick: int,
        snapshot_resolution: int,
        max_snapshots: Optional[int],
        additional_options: dict = None
    ) -> None:
        super(OncallRoutingBusinessEngine, self).__init__(
            scenario_name="oncall_routing",
            event_buffer=event_buffer,
            topology=topology,
            start_tick=start_tick,
            max_tick=max_tick,
            snapshot_resolution=snapshot_resolution,
            max_snapshots=max_snapshots,
            additional_options=additional_options
        )

        # Load config.
        self.update_config_root_path(__file__)
        if "config_path" in additional_options:
            self._config_path = additional_options["config_path"]
        print(f"Config path: {self._config_path}")

        with open(os.path.join(self._config_path, "config.yml")) as fp:
            self._config = convert_dottable(safe_load(fp))

        # Step 1: Init random seed
        random.seed(self._config.seed)

        # Step 2: Init oncall order generator, oncall order buffer, waiting order dict.
        print("Loading oncall orders.")
        self._oncall_order_generator = get_oncall_generator(self._config_path, self._config.data_loader_config)
        self._oncall_order_generator.reset()
        self._oncall_order_buffer: Deque[Order] = deque()
        self._waiting_order_dict: Dict[str, Order] = {}  # Orders already sent to agents and waiting for actions

        # Step 3: Init data loader, load route plan.
        print("Loading plans.")
        self._data_loader = get_data_loader(self._config_path, self._config.data_loader_config)
        remaining_plan: Dict[str, List[PlanElement]] = self._load_route_plan()

        # Step 4: Init predictor.
        self._actual_duration_predictor = ActualDurationSampler()
        self._estimated_duration_predictor = EstimatedDurationPredictor()

        # Step 5: Init Frame and snapshot.
        route_num = len(self._route_name_list)
        self._frame = gen_oncall_routing_frame(
            route_num=route_num,
            snapshots_num=self.calc_max_snapshots()
        )
        self._snapshots = self._frame.snapshots

        # Step 6: Init nodes (Carriers & Routes), name-index mapping.
        self._carriers: List[Carrier] = self._frame.carriers
        self._routes: List[Route] = self._frame.routes
        self._route_name2idx = {}
        self._carrier_name2idx = {}

        for idx, route_name in enumerate(self._route_name_list):
            self._route_name2idx[route_name] = idx

            carrier_name = f"C_{route_name}"
            self._carrier_name2idx[carrier_name] = idx

            carrier = self._carriers[idx]
            carrier.set_init_state(
                name=carrier_name,
                route_idx=idx,
                route_name=route_name,
                lat=self._config.station.latitude,
                lng=self._config.station.longitude,
                close_rtb=self._config.station.close_rtb,
            )

            route = self._routes[idx]
            route.set_init_state(
                name=route_name,
                carrier_idx=idx,
                carrier_name=carrier_name,
            )
            # Step 6-1: Init route plan.
            route.remaining_plan = remaining_plan[route_name]

            # Step 6-2: Init duration between stops
            for i in range(len(route.remaining_plan)):
                self._refresh_plan_duration(tick=-1, route_idx=route.idx, index=i)

        # Step 7: Create the carrier arrival events.
        self._load_carrier_arrival_event()

        # Step 8: Init Env metrics.
        self._init_metrics()

        # Register event handlers.
        self._register_events()

    def _load_route_plan(self) -> Dict[str, List[PlanElement]]:
        remaining_plan: Dict[str, List[PlanElement]] = self._data_loader.generate_plan()
        # TODO: sort the plan in a same location by (open time, close time)

        # TODO: fake head quarter order
        rtb_order = Order(
            order_id=str(next(GLOBAL_ORDER_COUNTER)),
            coordinate=Coordinate(lat=self._config.station.latitude, lng=self._config.station.longitude),
            open_time=self._config.data_loader_config.start_tick,
            close_time=self._config.data_loader_config.end_tick,
            is_delivery=None,
            status=OrderStatus.DUMMY
        )

        for plan in remaining_plan.values():
            plan.append(PlanElement(order=rtb_order))
        self._route_name_list: List[str] = sorted(list(remaining_plan.keys()))
        return remaining_plan

    def _load_carrier_arrival_event(self) -> None:
        for route in self._routes:
            if len(route.remaining_plan) > 0:
                carrier_arrival_payload = CarrierArrivalPayload(route.carrier_idx)
                carrier_arrival_event = self._event_buffer.gen_cascade_event(
                    tick=route.remaining_plan[0].actual_duration_from_last,
                    event_type=Events.CARRIER_ARRIVAL,
                    payload=carrier_arrival_payload
                )
                self._event_buffer.insert_event(carrier_arrival_event)

    def _init_metrics(self) -> None:
        self._total_oncall_num: int = 0
        self._unallocated_oncall_num: int = 0
        self._total_order_num: int = 0
        self._total_order_in_advance: int = 0
        self._total_order_delayed: int = 0
        self._total_order_delay_time: int = 0
        self._total_order_terminated: int = 0
        self._total_order_completed: int = 0

        for route in self._routes:
            self._total_order_num += len(route.remaining_plan) - 1

    def get_metrics(self) -> DocableDict:
        """Get current environment metrics information.

        Returns:
            DocableDict: Metrics information.
        """

        return DocableDict(
            metrics_desc,
            {
                "total_oncall_num": self._total_oncall_num,
                "unallocated_oncall_num": self._unallocated_oncall_num,
                "total_order_num": self._total_order_num,
                "total_order_in_advance": self._total_order_in_advance,
                "total_order_delayed": self._total_order_delayed,
                "total_order_delay_time": self._total_order_delay_time,
                "total_order_terminated": self._total_order_terminated,
                "total_order_completed": self._total_order_completed,
            }
        )

    @property
    def frame(self) -> FrameBase:
        return self._frame

    @property
    def snapshots(self) -> SnapshotList:
        return self._snapshots

    def get_agent_idx_list(self) -> List[int]:
        return list(range(len(self._routes)))

    def step(self, tick: int) -> None:
        # Update oncall orders
        oncall_orders = self._oncall_order_generator.get_oncall_orders(tick)
        if len(oncall_orders) > 0:
            oncall_receive_payload = OncallReceivePayload(oncall_orders)
            oncall_receive_event = self._event_buffer.gen_atom_event(
                tick=tick, event_type=Events.ONCALL_RECEIVE, payload=oncall_receive_payload
            )
            self._event_buffer.insert_event(oncall_receive_event)

        # Interrupt and throw decision event
        if (tick + 1) % self._config["interrupt_cycle"] == 0 and len(self._oncall_order_buffer) > 0:
            decision_event = self._event_buffer.gen_decision_event(
                tick=tick,
                payload=OncallRoutingPayload(oncall_orders=list(self._oncall_order_buffer))
            )
            self._event_buffer.insert_event(decision_event)

            self._waiting_order_dict = {order.id: order for order in self._oncall_order_buffer}
            self._oncall_order_buffer.clear()

    @property
    def configs(self) -> dict:
        return self._config

    def _reset_nodes(self, remaining_plan: Dict[str, List[PlanElement]]):
        for route in self._routes:
            route.reset()
            # Step 6-1: Initialize route plan
            route.remaining_plan = remaining_plan[route.name]
            # Step 6-2: Initialize duration between stops
            for i in range(len(route.remaining_plan)):
                self._refresh_plan_duration(tick=-1, route_idx=route.idx, index=i)

        for carrier in self._carriers:
            carrier.reset()

    def reset(self, keep_seed: bool = False) -> None:
        # Step 1
        new_seed = self._config.seed if keep_seed else random[GLOBAL_RAND_KEY].randint(0, 4095)
        random.seed(new_seed)

        # Step 2
        # TODO: replaced with a reset method
        GLOBAL_ORDER_COUNTER = count()
        self._oncall_order_generator.reset()
        self._oncall_order_buffer.clear()
        self._waiting_order_dict.clear()

        # Step 3
        # TODO: data_loader reset?
        remaining_plan = self._load_route_plan()

        # Step 4
        # TODO: predictor reset

        # Step 5
        self._snapshots.reset()
        self._frame.reset()

        # Step 6
        self._reset_nodes(remaining_plan=remaining_plan)

        # Step 7
        self._load_carrier_arrival_event()

        # Step 8
        self._init_metrics()

    def _refresh_plan_duration(self, tick: int, route_idx: int, index: int = 0) -> None:
        carrier_idx = self._routes[route_idx].carrier_idx
        plan = self._routes[route_idx].remaining_plan
        source_coord = self._carriers[carrier_idx].coordinate if index == 0 else plan[index - 1].order.coord
        target_coord = plan[index].order.coord

        estimated_duration = self._estimated_duration_predictor.predict(tick, source_coord, target_coord)
        actual_duration = self._actual_duration_predictor.sample(tick, source_coord, target_coord, estimated_duration)
        plan[index].actual_duration_from_last = actual_duration
        plan[index].estimated_duration_from_last = estimated_duration

    def _register_events(self) -> None:
        register_handler = self._event_buffer.register_event_handler

        register_handler(Events.ONCALL_RECEIVE, self._on_oncall_receive)
        register_handler(Events.CARRIER_ARRIVAL, self._on_carrier_arrival)
        register_handler(Events.ORDER_PROCESSING, self._on_order_processing)
        register_handler(Events.CARRIER_DEPARTURE, self._on_carrier_departure)
        register_handler(MaroEvents.TAKE_ACTION, self._on_action_received)

    def _on_oncall_receive(self, event: AtomEvent) -> None:
        payload = event.payload
        assert isinstance(payload, OncallReceivePayload)
        self._oncall_order_buffer.extend(payload.orders)
        self._total_oncall_num += len(payload.orders)
        self._total_order_num += len(payload.orders)

    def _on_carrier_arrival(self, event: CascadeEvent) -> None:
        payload = event.payload
        assert isinstance(payload, CarrierArrivalPayload)

        carrier_idx = payload.carrier_idx
        route_idx = self._carriers[carrier_idx].route_idx
        plan = self._routes[route_idx].remaining_plan

        # Update the location of the carrier.
        self._carriers[carrier_idx].in_stop = 1
        self._carriers[carrier_idx].update_coordinate(coord=plan[0].order.coord)

        if len(plan) == 0:
            return

        order_status = plan[0].order.get_status(event.tick, self._config.order_transition)

        # TODO: further simplify the order processing logic?
        if order_status == OrderStatus.NOT_READY:
            # Wait, and process when it's ready in the future.
            order_processing_payload = OrderProcessingPayload(carrier_idx=carrier_idx)
            order_processing_event = self._event_buffer.gen_cascade_event(
                tick=plan[0].order.open_time - self._config.order_transition.buffer_before_open_time,
                event_type=Events.ORDER_PROCESSING,
                payload=order_processing_payload
            )
            self._event_buffer.insert_event(order_processing_event)

        else:
            # Process right now
            order_processing_payload = OrderProcessingPayload(carrier_idx=carrier_idx)
            order_processing_event = self._event_buffer.gen_cascade_event(
                tick=event.tick,
                event_type=Events.ORDER_PROCESSING,
                payload=order_processing_payload
            )
            event.add_immediate_event(event=order_processing_event)

    def _on_order_processing(self, event: CascadeEvent) -> None:
        payload = event.payload
        assert isinstance(payload, OrderProcessingPayload)

        carrier_idx = payload.carrier_idx
        carrier = self._carriers[carrier_idx]
        route_idx = self._carriers[carrier_idx].route_idx
        plan = self._routes[route_idx].remaining_plan

        # Handle the orders can be finished.
        processing_time = 0
        processing_order = False
        while len(plan) > 0 and plan[0].order.coord == carrier.coordinate:
            order: Order = plan[0].order
            order_status = order.get_status(event.tick + processing_time, self._config.order_transition)

            # Update performance statistics.
            if order_status == OrderStatus.DUMMY:
                # TODO: if any logic for DUMMY
                plan.pop(0)
                continue

            elif order_status == OrderStatus.NOT_READY:
                # Wait, and process when it's ready in the future.
                order_processing_payload = OrderProcessingPayload(carrier_idx=carrier_idx)
                order_processing_event = self._event_buffer.gen_cascade_event(
                    tick=order.open_time - self._config.order_transition.buffer_before_open_time,
                    event_type=Events.ORDER_PROCESSING,
                    payload=order_processing_payload
                )
                self._event_buffer.insert_event(order_processing_event)
                return

            elif order_status == OrderStatus.COMPLETED:
                print(f"{event.tick + processing_time} Order with wrong status in order processing: {order}, skip")
                plan.pop(0)
                continue

            elif order_status == OrderStatus.TERMINATED:
                self._total_order_terminated += 1
                plan.pop(0)
                continue

            elif order_status == OrderStatus.READY_IN_ADVANCE:
                self._total_order_in_advance += 1
                carrier.in_advance_order_num += 1

            elif order_status == OrderStatus.IN_PROCESS_BUT_DELAYED:
                carrier.delayed_order_num += 1
                carrier.total_delayed_time += event.tick + processing_time - order.close_time
                self._total_order_delayed += 1
                self._total_order_delay_time += event.tick + processing_time - order.close_time

            self._total_order_completed += 1
            order.set_status(OrderStatus.COMPLETED)
            # TODO: to save finished order or not?

            # Update carrier payload information.
            # TODO
            # payload_factor = -1 if order.is_delivery else 1
            # carrier.payload_quantity += payload_factor * order.package_num
            # carrier.payload_volume += payload_factor * order.volume
            # carrier.payload_weight += payload_factor * order.weight

            processing_order = True
            if self._config.order_transition.processing_proportion_to_quantity:
                processing_time += self._config.order_transition.processing_time

            plan.pop(0)

        if processing_order and not self._config.order_transition.processing_proportion_to_quantity:
            processing_time = self._config.order_transition.processing_time

        # Add carrier departure event.
        if len(plan) > 0:
            departure_tick = event.tick + processing_time
            carrier_departure_payload = CarrierDeparturePayload(carrier_idx)
            carrier_departure_event = self._event_buffer.gen_atom_event(
                tick=departure_tick,
                event_type=Events.CARRIER_DEPARTURE,
                payload=carrier_departure_payload
            )
            if departure_tick == event.tick:
                event.add_immediate_event(carrier_departure_event)
            else:
                self._event_buffer.insert_event(carrier_departure_event)

    def _on_carrier_departure(self, event: AtomEvent) -> None:
        payload = event.payload
        assert isinstance(payload, CarrierDeparturePayload)

        carrier_idx = payload.carrier_idx
        route_idx = self._carriers[carrier_idx].route_idx

        self._carriers[carrier_idx].in_stop = 0

        plan = self._routes[route_idx].remaining_plan
        # Add next carrier arrival event.
        if len(plan) > 0:
            carrier_arrival_payload = CarrierArrivalPayload(carrier_idx)
            carrier_arrival_event = self._event_buffer.gen_cascade_event(
                tick=event.tick + plan[0].actual_duration_from_last,
                event_type=Events.CARRIER_ARRIVAL,
                payload=carrier_arrival_payload
            )
            self._event_buffer.insert_event(carrier_arrival_event)

    def _on_action_received(self, event: CascadeEvent) -> None:
        actions = event.payload
        assert isinstance(actions, list)

        # Aggregate actions by route
        action_by_route: Dict[str, List[Action]] = defaultdict(list)
        for action in actions:
            assert isinstance(action, Action)
            action_by_route[action.route_name].append(action)

        for route_name, actions in action_by_route.items():
            # Sort actions by: 1) insert index, 2) in-segment order
            actions.sort(key=lambda _action: (_action.insert_index, _action.in_segment_order))

            route_idx = self._route_name2idx[route_name]
            old_plan = self._routes[route_idx].remaining_plan
            new_plan = []
            refresh_indexes = []
            j = 0
            for i, old_plan_element in enumerate(old_plan):
                has_new_plan = False
                # Insert all oncall orders that should be inserted before this old stop
                while j < len(actions) and actions[j].insert_index < i:
                    new_order_id = actions[j].order_id
                    new_order = self._waiting_order_dict.pop(new_order_id)  # Remove this order from waiting dict
                    new_plan_element = PlanElement(order=new_order)
                    new_plan.append(new_plan_element)
                    refresh_indexes.append(len(new_plan) - 1)
                    j += 1
                    has_new_plan = True

                # Insert this old stop.
                new_plan.append(old_plan_element)

                # If there are new oncall orders before this old stop, refresh predicted time.
                if has_new_plan:
                    refresh_indexes.append(len(new_plan) - 1)

            self._routes[route_idx].remaining_plan = new_plan
            for index in refresh_indexes:
                self._refresh_plan_duration(tick=event.tick, route_idx=route_idx, index=index)

        # Put back suspended oncall orders
        self._oncall_order_buffer = deque([order for order in self._waiting_order_dict.values()])
        self._waiting_order_dict.clear()

    def post_step(self, tick: int) -> bool:
        is_done: bool = (tick + 1 == self._max_tick)
        self._unallocated_oncall_num = len(self._oncall_order_buffer)
        # TODO: handle the orders left issue
        if is_done:
            for route in self._routes:
                plan = route.remaining_plan
                if len(plan) == 1:
                    print(
                        f"carrier_idx: {route.carrier_idx}, "
                        f"remaining plan: {len(route.remaining_plan)} {plan[0].order}"
                    )
                elif len(plan) > 0:
                    print(f"carrier_idx: {route.carrier_idx}, remaining plan: {len(route.remaining_plan)}")
        return is_done
