# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from collections import defaultdict, deque
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
from .common import Action, CarrierArrivalPayload, Events, OncallReceivePayload, OncallRoutingPayload, PlanElement
from .coordinate import Coordinate
from .duration_time_predictor import ActualDurationSampler, EstimatedDurationPredictor
from .frame_builder import gen_oncall_routing_frame
from .order import GLOBAL_ORDER_COUNTER, Order, OrderStatus
from .route import Route
from .utils import GLOBAL_RAND_KEY

metrics_desc = """
Oncall routing metrics used provide statistics information until now.
It contains following keys:

total_order_num(int): The total number of orders.
total_failures_due_to_early_arrival(int):
total_order_in_advance(int):
total_order_delayed(int): The total delayed order quantity of all carriers.
total_order_delay_time(int): The total order delay time of all carriers.
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

        # Load config
        self.update_config_root_path(__file__)
        if "config_path" in additional_options:
            self._config_path = additional_options["config_path"]
        print(f"Config path: {self._config_path}")

        with open(os.path.join(self._config_path, "config.yml")) as fp:
            self._config = convert_dottable(safe_load(fp))

        # Init random seed
        random.seed(self._config.seed)

        self._init_metrics()

        # Init oncall order generator
        print("Loading oncall orders.")
        self._oncall_order_generator = get_oncall_generator(self._config_path)
        self._oncall_order_generator.reset()
        self._oncall_order_buffer: Deque[Order] = deque()
        print("Oncall orders loaded.")

        self._waiting_order_dict: Dict[str, Order] = {}  # Orders already sent to agents and waiting for actions

        self._actual_duration_predictor = ActualDurationSampler()
        self._estimated_duration_predictor = EstimatedDurationPredictor()

        # ##### Load plan #####
        print("Loading plans.")
        data_loader = get_data_loader(self._config_path)
        remaining_plan: Dict[str, List[PlanElement]] = data_loader.generate_plan()

        # TODO: fake head quarter order
        rtb_order = Order(
            order_id=str(next(GLOBAL_ORDER_COUNTER)),
            coordinate=Coordinate(lat=self._config.station.latitude, lng=self._config.station.longitude),
            open_time=0,
            close_time=1440 - 1
        )

        for plan in remaining_plan.values():
            plan.append(PlanElement(order=rtb_order))
        route_name_list = sorted(list(remaining_plan.keys()))
        print("Plans loaded.")

        # Initialize Frames.
        route_num = len(route_name_list)
        self._frame = gen_oncall_routing_frame(
            route_num=route_num,
            snapshots_num=self.calc_max_snapshots()
        )
        self._snapshots = self._frame.snapshots

        # Initialize Carriers & Routes.
        self._carriers: List[Carrier] = self._frame.carriers
        self._routes: List[Route] = self._frame.routes
        self._route_name2idx = {}
        self._carrier_name2idx = {}

        for idx, route_name in enumerate(route_name_list):
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
            route.remaining_plan = remaining_plan[route_name]
            self._total_order_num += len(remaining_plan[route_name]) - 1

        # Initialize duration between stops and create the carrier arrival events.
        for route in self._routes:
            for i in range(len(route.remaining_plan)):
                self._refresh_plan_duration(tick=-1, route_idx=route.idx, index=i)

            if len(route.remaining_plan) > 0:
                carrier_arrival_payload = CarrierArrivalPayload(route.carrier_idx)
                carrier_arrival_event = self._event_buffer.gen_atom_event(
                    tick=route.remaining_plan[0].actual_duration_from_last,
                    event_type=Events.CARRIER_ARRIVAL,
                    payload=carrier_arrival_payload
                )
                self._event_buffer.insert_event(carrier_arrival_event)

        self._register_events()

    def _init_metrics(self) -> None:
        self._total_order_num: int = 0
        self._failed_due_to_early_arrival: int = 0
        self._total_order_in_advance: int = 0
        self._total_order_delayed: int = 0
        self._total_order_delay_time: int = 0
        self._total_order_completed: int = 0

    def _get_metrics(self) -> DocableDict:
        """Get current environment metrics information.

        Returns:
            DocableDict: Metrics information.
        """

        return DocableDict(
            metrics_desc,
            {
                "total_order_num": self._total_order_num,
                "total_failures_due_to_early_arrival": self._failed_due_to_early_arrival,
                "total_order_in_advance": self._total_order_in_advance,
                "total_order_delayed": self._total_order_delayed,
                "total_order_delay_time": self._total_order_delay_time,
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

    def reset(self, keep_seed: bool = False) -> None:
        new_seed = self._config.seed if keep_seed else random[GLOBAL_RAND_KEY].randint(0, 4095)
        random.seed(new_seed)
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
        register_handler(MaroEvents.TAKE_ACTION, self._on_action_received)

    def _on_oncall_receive(self, event: AtomEvent) -> None:
        payload = event.payload
        assert isinstance(payload, OncallReceivePayload)
        self._oncall_order_buffer.extend(payload.orders)
        self._total_order_num += len(payload.orders)

    def _on_carrier_arrival(self, event: AtomEvent) -> None:
        payload = event.payload
        assert isinstance(payload, CarrierArrivalPayload)

        carrier_idx = payload.carrier_idx
        route_idx = self._carriers[carrier_idx].route_idx

        # Handle the orders can be finished.
        plan = self._routes[route_idx].remaining_plan
        cur_arrival: PlanElement = plan.pop(0)
        self._order_processing(self._carriers[carrier_idx], cur_arrival.order, event.tick)
        while len(plan) > 0 and plan[0].actual_duration_from_last == 0:  # TODO: or compare the coordinate?
            cur_arrival = plan.pop(0)
            self._order_processing(self._carriers[carrier_idx], cur_arrival.order, event.tick)

        self._carriers[carrier_idx].update_coordinate(cur_arrival.order.coord)

        # Add next arrival event.
        if len(plan) > 0:
            carrier_arrival_payload = CarrierArrivalPayload(carrier_idx)
            carrier_arrival_event = self._event_buffer.gen_atom_event(
                tick=event.tick + plan[0].actual_duration_from_last,
                event_type=Events.CARRIER_ARRIVAL,
                payload=carrier_arrival_payload
            )
            self._event_buffer.insert_event(carrier_arrival_event)

    def _order_processing(self, carrier: Carrier, order: Order, tick: int) -> None:
        order_status = order.get_status(tick, self._config.order_transition.buffer_before_open_time)

        # Update performance statistics.
        if order_status == OrderStatus.NOT_READY:
            self._failed_due_to_early_arrival += 1
            order.set_status(OrderStatus.TERMINATED)
            return
        elif order_status == OrderStatus.FINISHED or order_status == OrderStatus.TERMINATED:
            print(f"Order with wrong status in order processing: {order}, skip")
            return
        elif order_status == OrderStatus.READY_IN_ADVANCE:
            self._total_order_in_advance += 1
            carrier.in_advance_order_num += 1
        elif order_status == OrderStatus.IN_PROCESS_BUT_DELAYED:
            carrier.delayed_order_num += 1
            carrier.total_delayed_time += tick - order.close_time
            self._total_order_delayed += 1
            self._total_order_delay_time += tick - order.close_time

        order.set_status(OrderStatus.FINISHED)
        self._total_order_completed += 1

        # Update carrier payload information.
        # TODO
        # payload_factor = -1 if order.is_delivery else 1
        # carrier.payload_quantity += payload_factor * order.package_num
        # carrier.payload_volume += payload_factor * order.volume
        # carrier.payload_weight += payload_factor * order.weight

        # TODO: to save finished order or not?
        # TODO: add order processing time

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
        return tick + 1 == self._max_tick
