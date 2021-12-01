# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional

from maro.backends.frame import FrameBase, SnapshotList
from yaml import safe_load

from maro.data_lib.oncall_routing import FromHistoryOncallOrderGenerator
from maro.data_lib.oncall_routing.data_loader import FromHistoryPlanLoader, PlanLoader, SamplePlanLoader
from maro.data_lib.oncall_routing.oncall_order_generator import OncallOrderGenerator, SampleOncallOrderGenerator
from maro.event_buffer import AtomEvent, CascadeEvent, EventBuffer, MaroEvents
from maro.simulator.scenarios import AbsBusinessEngine
from maro.simulator.utils import random
from maro.utils import convert_dottable
from . import Coordinate
from .arrival_time_predictor import ActualDurationSampler, EstimatedDurationPredictor
from .carrier import Carrier
from .common import Action, CarrierArrivalPayload, Events, OncallReceivePayload, OncallRoutingPayload, PlanElement
from .frame_builder import gen_oncall_routing_frame
from .order import GLOBAL_ORDER_COUNTER, Order
from .route import Route
from .utils import GLOBAL_RAND_KEY


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

        # TODO: fake head quarter order
        self._rtb_order = Order()
        self._rtb_order.id = str(next(GLOBAL_ORDER_COUNTER))
        self._rtb_order.coord = Coordinate(
            lat=self._config.station.latitude,
            lng=self._config.station.longitude
        )

        # Init random seed
        self._default_random_seed = 1024
        random.seed(self._default_random_seed)

        # Init oncall order generator
        print("Loading oncall orders.")
        self._oncall_order_generator = self._get_oncall_generator()
        self._oncall_order_generator.reset()
        self._oncall_order_buffer: Deque[Order] = deque()
        print("Oncall orders loaded.")

        self._waiting_order_dict: Dict[str, Order] = {}  # Orders already sent to agents and waiting for actions

        self._actual_duration_predictor = ActualDurationSampler()
        self._estimated_duration_predictor = EstimatedDurationPredictor()

        # ##### Load plan #####
        print("Loading plans.")
        data_loader = self._get_data_loader()
        self._remain_plan: Dict[str, List[PlanElement]] = data_loader.generate_plan()
        for plan in self._remain_plan.values():  # TODO
            plan.append(PlanElement(order=self._rtb_order))

        # Initialize Frames.
        route_num = len(self._remain_plan.keys())
        self._frame = gen_oncall_routing_frame(
            route_num=route_num,
            snapshots_num=self.calc_max_snapshots()
        )
        self._snapshots = self._frame.snapshots

        # Initialize Carriers & Routes.
        self._carriers: List[Carrier] = self._frame.carriers
        self._routes: List[Route] = self._frame.routes
        self._route_name2idx = {}

        for idx, route_name in enumerate(sorted(list(self._remain_plan.keys()))):
            self._route_name2idx[route_name] = idx
            carrier = self._carriers[idx]
            route = self._routes[idx]

            carrier_name = f"C_{route_name}"

            carrier.set_init_state(
                name=carrier_name,
                route_idx=idx,
                route_name=route_name,
                lat=self._config.station.latitude,
                lng=self._config.station.longitude,
                close_rtb=self._config.station.close_rtb,
            )

            route.set_init_state(
                name=route_name,
                carrier_idx=idx,
                carrier_name=carrier_name,
            )
            route.remaining_plan = self._remain_plan[route_name]

        # Initialize Arrival Time.
        for route in self._routes:
            for i in range(len(route.remaining_plan)):
                self._refresh_arr_time(tick=-1, route_idx=route.idx, index=i)

        self._upcoming_arr_time: Dict[str, Optional[int]] = {
            route_name: plan[0].actual_duration_from_last for route_name, plan in self._remain_plan.items()
        }
        print("Plans loaded.")

        self._register_events()

    def _get_oncall_generator(self) -> OncallOrderGenerator:
        if os.path.exists(os.path.join(self._config_path, "oncall_orders.csv")):
            return FromHistoryOncallOrderGenerator(os.path.join(self._config_path, "oncall_orders.csv"))
        if os.path.exists(os.path.join(self._config_path, "oncall_info.yml")):
            return SampleOncallOrderGenerator(self._config_path)
        raise ValueError("Cannot found correct oncall data.")

    def _get_data_loader(self) -> PlanLoader:
        if os.path.exists(os.path.join(self._config_path, "routes.csv")):
            return FromHistoryPlanLoader(os.path.join(self._config_path, "routes.csv"))
        if os.path.exists(os.path.join(self._config_path, "route_coord.txt")):
            return SamplePlanLoader(self._config_path)
        raise ValueError("Cannot found correct route data.")

    @property
    def frame(self) -> FrameBase:
        return self._frame

    @property
    def snapshots(self) -> SnapshotList:
        return self._snapshots

    def get_agent_idx_list(self) -> List[int]:
        return list(range(len(self._routes)))

    def step(self, tick: int) -> None:
        # Carrier arrive its next destination
        for route_name in self._route_name2idx.keys():
            if self._upcoming_arr_time[route_name] is not None and self._upcoming_arr_time[route_name] == tick:
                carrier_arrival_payload = CarrierArrivalPayload(route_name)
                carrier_arrival_event = self._event_buffer.gen_atom_event(
                    tick=tick, event_type=Events.CARRIER_ARRIVAL, payload=carrier_arrival_payload
                )
                self._event_buffer.insert_event(carrier_arrival_event)

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
                payload=OncallRoutingPayload(
                    oncall_orders=list(self._oncall_order_buffer)
                )
            )
            self._event_buffer.insert_event(decision_event)

            self._waiting_order_dict = {order.id: order for order in self._oncall_order_buffer}
            self._oncall_order_buffer.clear()

    @property
    def configs(self) -> dict:
        return self._config

    def reset(self, keep_seed: bool = False) -> None:
        new_seed = self._default_random_seed if keep_seed else random[GLOBAL_RAND_KEY].randint(0, 4095)
        random.seed(new_seed)

    def _refresh_arr_time(self, tick: int, route_idx: int, index: int = 0) -> None:
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

    def _on_carrier_arrival(self, event: AtomEvent) -> None:
        payload = event.payload
        assert isinstance(payload, CarrierArrivalPayload)

        route_name = payload.route_name
        route_idx = self._route_name2idx[route_name]
        carrier_idx = self._routes[route_idx].carrier_idx

        # TODO: deliver / pickup packages

        plan = self._routes[route_idx].remaining_plan
        cur_arrival = plan.pop(0)  # Finish the current plan

        self._carriers[carrier_idx].update_coordinate(cur_arrival.order.coord)
        self._upcoming_arr_time[route_name] = None if len(plan) == 0 else event.tick + plan[0].actual_duration_from_last

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
                self._refresh_arr_time(tick=event.tick, route_idx=route_idx, index=index)

        # Put back suspended oncall orders
        self._oncall_order_buffer = deque([order for order in self._waiting_order_dict.values()])
        self._waiting_order_dict.clear()

    def post_step(self, tick: int) -> bool:
        return all(len(plan) == 0 for plan in self._remain_plan.values())
