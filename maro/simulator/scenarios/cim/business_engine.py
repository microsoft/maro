# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import os
from math import ceil, floor

import numpy as np
from yaml import safe_load

from maro.backends.frame import FrameBase, SnapshotList
from maro.data_lib.cim import CimDataContainerWrapper, Order, Stop
from maro.event_buffer import AtomEvent, CascadeEvent, EventBuffer, MaroEvents
from maro.simulator.scenarios import AbsBusinessEngine
from maro.simulator.scenarios.helpers import DocableDict
from maro.simulator.scenarios.matrix_accessor import MatrixAttributeAccessor
from maro.streamit import streamit

from .common import Action, ActionScope, ActionType, DecisionEvent
from .event_payload import EmptyReturnPayload, LadenReturnPayload, VesselDischargePayload, VesselStatePayload
from .events import Events
from .frame_builder import gen_cim_frame
from .ports_order_export import PortOrderExporter

metrics_desc = """
CIM metrics used provide statistics information until now (may be in the middle of current tick).
It contains following keys:

order_requirements (int): Accumulative orders until now.
container_shortage (int): Accumulative shortage until now.
operation_number (int): Total empty operation (both load and discharge) cost,
    the cost factors can be configured through 'load_cost_factor' and 'dsch_cost_factor' in configuration file.
"""


class CimBusinessEngine(AbsBusinessEngine):
    """Cim business engine, used simulate CIM related problem."""

    def __init__(
        self, event_buffer: EventBuffer, topology: str, start_tick: int, max_tick: int,
        snapshot_resolution: int, max_snapshots: int, additional_options: dict = None
    ):
        super().__init__(
            "cim", event_buffer, topology, start_tick, max_tick,
            snapshot_resolution, max_snapshots, additional_options
        )

        # Update self._config_path with current file path.
        self.update_config_root_path(__file__)

        # Load data from wrapper.
        self._data_cntr: CimDataContainerWrapper = CimDataContainerWrapper(
            self._config_path, max_tick, self._topology
        )

        # Create a copy of config object to expose to others, and not affect generator.
        self._config = {}
        config_path = os.path.join(self._config_path, "config.yml")
        if os.path.exists(config_path):
            with open(config_path) as fp:
                self._config = safe_load(fp)

        self._vessels = []
        self._ports = []
        self._frame = None
        self._full_on_ports: MatrixAttributeAccessor = None
        self._full_on_vessels: MatrixAttributeAccessor = None
        self._vessel_plans: MatrixAttributeAccessor = None
        self._port_orders_exporter = PortOrderExporter("enable-dump-snapshot" in additional_options)

        self._load_cost_factor: float = self._data_cntr.load_cost_factor
        self._dsch_cost_factor: float = self._data_cntr.dsch_cost_factor

        # Used to collect total cost to avoid to much snapshot querying.
        self._total_operate_num: float = 0

        self._init_frame()

        # Snapshot list should be initialized after frame.
        self._snapshots = self._frame.snapshots

        self._register_events()

        # As we already unpack the route to the max tick, we can insert all departure events at the beginning.
        self._load_departure_events()

        # Since there is no Arrival Event at the very beginning, init the vessel states maunally.
        self._init_vessel_plans()

        self._stream_base_info()

    @property
    def configs(self):
        """dict: Configurations of CIM business engine."""
        return self._config

    @property
    def frame(self) -> FrameBase:
        """FrameBase: Frame of current business engine."""
        return self._frame

    @property
    def snapshots(self) -> SnapshotList:
        """SnapshotList: Snapshot list of current frame."""
        return self._snapshots

    def step(self, tick: int):
        """Called at each tick to generate orders and arrival events.

        Args:
            tick (int): Tick to generate orders.
        """

        # At each tick:
        # 1. Generate orders for this tick.
        # 2. Transfer orders into events (ORDER).
        # 3. Check and add vessel arrival event (atom and cascade).

        total_empty_number = sum(
            [node.empty for node in self._ports + self._vessels])

        for order in self._data_cntr.get_orders(tick, total_empty_number):
            # Use cascade event to support insert sub events.
            order_evt = self._event_buffer.gen_cascade_event(tick, Events.ORDER, order)

            self._event_buffer.insert_event(order_evt)
            self._port_orders_exporter.add(order)

        # Used to hold decision event of this tick, we will append this at the end
        # to make sure all the other logic finished.
        # TODO: Remove it after event priority is supported.
        decision_evt_list = []

        for vessel in self._vessels:
            vessel_idx: int = vessel.idx
            loc_idx: int = vessel.next_loc_idx

            stop: Stop = self._data_cntr.vessel_stops[vessel_idx, loc_idx]
            port_idx: int = stop.port_idx

            # At the beginning the vessel is parking at port, will not invoke arrive event.
            if loc_idx > 0:
                # Check if there is any arrive event.
                if stop.arrival_tick == tick:
                    arrival_payload = VesselStatePayload(port_idx, vessel_idx)

                    # This vessel will arrive at current tick.
                    arrival_event = self._event_buffer.gen_atom_event(
                        tick, Events.VESSEL_ARRIVAL, arrival_payload)

                    # Then it will load full.
                    load_event = self._event_buffer.gen_atom_event(
                        tick, Events.LOAD_FULL, arrival_payload)

                    self._event_buffer.insert_event(arrival_event)
                    self._event_buffer.insert_event(load_event)

                    # Generate cascade event and payload.
                    decision_payload = DecisionEvent(
                        tick, port_idx, vessel_idx, self.snapshots, self.action_scope, self.early_discharge
                    )

                    decision_event: CascadeEvent = self._event_buffer.gen_decision_event(tick, decision_payload)

                    decision_evt_list.append(decision_event)

            if loc_idx > 0 and stop.arrival_tick == tick:
                self._vessel_plans[vessel_idx, port_idx] = stop.arrival_tick

        # Insert the cascade events at the end.
        for event in decision_evt_list:
            self._event_buffer.insert_event(event)

    def post_step(self, tick: int):
        """Post-process after each step.

        Args:
            tick (int): Tick to process.
        """
        self._stream_data()

        if (tick + 1) % self._snapshot_resolution == 0:
            # Update acc_fulfillment before take snapshot.
            for port in self._ports:
                port.acc_fulfillment = port.acc_booking - port.acc_shortage

            # Before go to next tick, we will take a snapshot first.
            self._frame.take_snapshot(self.frame_index(tick))

            # Reset port statistics (by tick) fields.
            for port in self._ports:
                port.shortage = 0
                port.booking = 0
                port.fulfillment = 0
                port.transfer_cost = 0

        return tick + 1 == self._max_tick

    def reset(self):
        """Reset the business engine, it will reset frame value."""

        self._snapshots.reset()

        self._frame.reset()

        self._reset_nodes()

        self._data_cntr.reset()

        # Insert departure event again.
        self._load_departure_events()

        self._total_operate_num = 0

    def action_scope(self, port_idx: int, vessel_idx: int) -> ActionScope:
        """Get the action scope of specified agent.

        Args:
            port_idx (int): Index of specified agent.
            vessel_idx (int): Index of specified vessel to take the action.

        Returns:
            ActionScope: Contains load and discharge scope.
        """
        port = self._ports[port_idx]
        vessel = self._vessels[vessel_idx]

        return ActionScope(load=min(port.empty, vessel.remaining_space), discharge=vessel.empty)

    def early_discharge(self, vessel_idx: int) -> int:
        """Get the early discharge number of specified vessel.

        Args:
            vessel_idx (int): Index of specified vessel.
        """
        return self._vessels[vessel_idx].early_discharge

    def get_metrics(self) -> DocableDict:
        """Get metrics information for cim scenario.

        Args:
            dict: A dict that contains "perf", "total_shortage" and "total_cost",
                and can use help method to show help docs.
        """
        total_shortage = sum([p.acc_shortage for p in self._ports])
        total_booking = sum([p.acc_booking for p in self._ports])

        return DocableDict(
            metrics_desc,
            order_requirements=total_booking,
            container_shortage=total_shortage,
            operation_number=self._total_operate_num
        )

    def get_node_mapping(self) -> dict:
        """Get node name mappings related with this environment.

        Returns:
            dict: Node name to index mapping dictionary.
        """
        return {
            "ports": self._data_cntr.port_mapping,
            "vessels": self._data_cntr.vessel_mapping
        }

    def get_event_payload_detail(self) -> dict:
        """dict: Event payload details of current scenario."""
        return {
            Events.ORDER.name: Order.summary_key,
            Events.RETURN_FULL.name: LadenReturnPayload.summary_key,
            Events.VESSEL_ARRIVAL.name: VesselStatePayload.summary_key,
            Events.LOAD_FULL.name: VesselStatePayload.summary_key,
            Events.DISCHARGE_FULL.name: VesselDischargePayload.summary_key,
            Events.PENDING_DECISION.name: DecisionEvent.summary_key,
            Events.LOAD_EMPTY.name: Action.summary_key,
            Events.DISCHARGE_EMPTY.name: Action.summary_key,
            Events.VESSEL_DEPARTURE.name: VesselStatePayload.summary_key,
            Events.RETURN_EMPTY.name: EmptyReturnPayload.summary_key
        }

    def get_agent_idx_list(self) -> list:
        """Get port index list related with this environment.

        Returns:
            list: A list of port index.
        """
        return [i for i in range(self._data_cntr.port_number)]

    def dump(self, folder: str):
        self._port_orders_exporter.dump(folder)

    def _init_nodes(self):
        # Init ports.
        for port_settings in self._data_cntr.ports:
            port = self._ports[port_settings.index]
            port.set_init_state(port_settings.name,
                                port_settings.capacity, port_settings.empty)

        # Init vessels.
        for vessel_setting in self._data_cntr.vessels:
            vessel = self._vessels[vessel_setting.index]

            vessel.set_init_state(
                vessel_setting.name,
                self._data_cntr.container_volume,
                vessel_setting.capacity,
                self._data_cntr.route_mapping[vessel_setting.route_name],
                vessel_setting.empty
            )

        # Init vessel plans.
        self._vessel_plans[:] = -1

    def _reset_nodes(self):
        # Reset both vessels and ports.
        # NOTE: This should be called after frame.reset.
        for port in self._ports:
            port.reset()

        for vessel in self._vessels:
            vessel.reset()

        # Reset vessel plans.
        self._vessel_plans[:] = -1

    def _register_events(self):
        """Register events."""
        register_handler = self._event_buffer.register_event_handler

        register_handler(Events.RETURN_FULL, self._on_full_return)
        register_handler(Events.RETURN_EMPTY, self._on_empty_return)
        register_handler(Events.ORDER, self._on_order_generated)
        register_handler(Events.LOAD_FULL, self._on_full_load)
        register_handler(Events.VESSEL_ARRIVAL, self._on_arrival)
        register_handler(Events.VESSEL_DEPARTURE, self._on_departure)
        register_handler(Events.DISCHARGE_FULL, self._on_discharge)
        register_handler(MaroEvents.TAKE_ACTION, self._on_action_received)

    def _load_departure_events(self):
        """Insert leaving event at the beginning as we already unpack the root to a loop at the beginning."""

        for vessel_idx, stops in enumerate(self._data_cntr.vessel_stops[:]):
            for stop in stops:
                payload = VesselStatePayload(stop.port_idx, vessel_idx)
                dep_evt = self._event_buffer.gen_atom_event(stop.leave_tick, Events.VESSEL_DEPARTURE, payload)

                self._event_buffer.insert_event(dep_evt)

    def _init_vessel_plans(self):
        for vessel in self._vessels:
            vessel.is_parking = 1 if vessel.last_loc_idx == vessel.next_loc_idx else 0
            stop: Stop = self._data_cntr.vessel_stops[vessel.idx, vessel.last_loc_idx]
            vessel.loc_port_idx = stop.port_idx

            # Initialize the past and future stop list.
            past_stops = self._data_cntr.vessel_past_stops[vessel.idx, vessel.last_loc_idx, vessel.next_loc_idx]
            future_stops = self._data_cntr.vessel_future_stops[vessel.idx, vessel.last_loc_idx, vessel.next_loc_idx]
            vessel.set_stop_list(past_stops, future_stops)

            # Update the vessel plans.
            for plan_port_idx, plan_tick in self._data_cntr.vessel_planned_stops[
                vessel.idx, vessel.route_idx, vessel.last_loc_idx
            ]:
                self._vessel_plans[vessel.idx, plan_port_idx] = plan_tick

    def _init_frame(self):
        """Initialize the frame based on data generator."""
        port_num = self._data_cntr.port_number
        vessel_num = self._data_cntr.vessel_number
        stop_num = (self._data_cntr.past_stop_number,
                    self._data_cntr.future_stop_number)

        self._frame = gen_cim_frame(
            port_num, vessel_num, stop_num, self.calc_max_snapshots())

        self._ports = self._frame.ports
        self._vessels = self._frame.vessels

        self._full_on_ports = self._frame.matrix[0]["full_on_ports"]
        self._full_on_vessels = self._frame.matrix[0]["full_on_vessels"]
        self._vessel_plans = self._frame.matrix[0]["vessel_plans"]

        self._init_nodes()

    def _get_reachable_ports(self, vessel_idx: int):
        """Get ports that specified vessel can reach (for order), return a list of tuple (port_id, arrival_tick).

        Args:
            vessel_idx (int): Index of specified vessel.

        Returns:
            Reachable port index list of specified vessel.
        """
        vessel = self._vessels[vessel_idx]

        return self._data_cntr.reachable_stops[vessel_idx, vessel.route_idx, vessel.next_loc_idx]

    def _get_pending_full(self, src_port_idx: int, dest_port_idx: int):
        """Get pending full number from src_port_idx to dest_port_idx."""
        return self._full_on_ports[src_port_idx, dest_port_idx]

    def _set_pending_full(self, src_port_idx: int, dest_port_idx: int, value):
        """Set the full number from src_port_idx to dest_port_idx."""
        assert value >= 0

        self._full_on_ports[src_port_idx, dest_port_idx] = value

    def _on_order_generated(self, event: CascadeEvent):
        """When there is an order generated, we should do:
        1. Generate a LADEN_RETURN event by configured buffer time: \
        The event will be inserted to the immediate_event_list ASAP if the configured buffer time is 0, \
        else the event will be inserted to the event buffer directly.
        2. Update port state: on_shipper +, empty -.

        Args:
            event (CascadeEvent): Order event object.
        """
        order: Order = event.payload
        src_port = self._ports[order.src_port_idx]

        execute_qty = order.quantity
        src_empty = src_port.empty
        src_port.booking += execute_qty
        src_port.acc_booking += execute_qty

        # Check if there is any shortage.
        if src_empty < order.quantity:
            # Booking & shortage.
            shortage_qty = order.quantity - src_empty
            src_port.shortage += shortage_qty
            src_port.acc_shortage += shortage_qty
            execute_qty = src_empty

        # Update port state.
        src_port.empty -= execute_qty
        # Full contianers that pending to return.
        src_port.on_shipper += execute_qty

        buffer_ticks = self._data_cntr.full_return_buffers[src_port.idx]

        payload = LadenReturnPayload(
            src_port_idx=order.src_port_idx, dest_port_idx=order.dest_port_idx, quantity=execute_qty
        )

        laden_return_evt = self._event_buffer.gen_atom_event(
            tick=event.tick + buffer_ticks, event_type=Events.RETURN_FULL, payload=payload
        )

        # If buffer_tick is 0, we should execute it as this tick.
        if buffer_ticks == 0:
            event.add_immediate_event(laden_return_evt)
        else:
            self._event_buffer.insert_event(laden_return_evt)

    def _on_full_return(self, event: AtomEvent):
        """Handler for processing the event that full containers are returned from shipper.

        Once the full containers are returned, the containers are ready to be loaded. The workflow is:
        1. First move the container from on_shipper to full (update state: on_shipper -> full).
        2. Then append the container to the port pending list.
        """
        payload: LadenReturnPayload = event.payload

        src_port = self._ports[payload.src_port_idx]
        src_port.on_shipper -= payload.quantity
        src_port.full += payload.quantity

        pending_full_number = self._get_pending_full(
            payload.src_port_idx, payload.dest_port_idx)

        self._set_pending_full(
            payload.src_port_idx, payload.dest_port_idx, pending_full_number + payload.quantity)

    def _on_full_load(self, event: AtomEvent):
        """Handler for processing event that a vessel need to load full containers from current port.

        When there is a vessel arrive at a port:
        1. Discharge full (we ignore this action here, as we will generate a discharge event \
        after a vessel have loaded any full).
        2. Load full by destination id, and generate discharge event.
        3. Update vessel.state to PARKING.
        4. Fill future stop list.
        5. Early discharge.

        Args:
            event (AtomEvent): Arrival event object.
        """

        arrival_obj: VesselStatePayload = event.payload
        vessel_idx: int = arrival_obj.vessel_idx
        port_idx: int = arrival_obj.port_idx
        vessel = self._vessels[vessel_idx]
        port = self._ports[port_idx]
        container_volume = self._data_cntr.container_volume
        vessel_capacity = vessel.capacity

        # Update vessel state.
        vessel.last_loc_idx = vessel.next_loc_idx

        # NOTE: This remaining space do not contains empty, as we can early discharge them if no enough space.
        remaining_space = vessel_capacity - vessel.full * container_volume

        # How many containers we can load.
        acceptable_number = floor(remaining_space / container_volume)
        total_load_qty = 0

        for next_port_idx, arrival_tick in self._get_reachable_ports(vessel_idx):
            full_number_to_next_port = self._get_pending_full(
                port_idx, next_port_idx)

            if acceptable_number > 0 and full_number_to_next_port > 0:
                # We can load some full.
                loaded_qty = min(full_number_to_next_port, acceptable_number)
                total_load_qty += loaded_qty

                # Update port state.
                self._set_pending_full(
                    port_idx, next_port_idx, full_number_to_next_port - loaded_qty)

                port.full -= loaded_qty
                vessel.full += loaded_qty

                # Update state.
                self._full_on_vessels[vessel_idx, next_port_idx] += loaded_qty

                acceptable_number -= loaded_qty

                # Generate a discharge event, as we know when the vessel will arrive at destination.
                payload = VesselDischargePayload(vessel_idx, port_idx, next_port_idx, loaded_qty)
                dsch_event = self._event_buffer.gen_cascade_event(arrival_tick, Events.DISCHARGE_FULL, payload)

                self._event_buffer.insert_event(dsch_event)

        # Early discharge.
        total_container = vessel.full + vessel.empty

        vessel.early_discharge = 0

        if total_container * container_volume > vessel.capacity:
            early_discharge_number = \
                total_container - ceil(vessel.capacity / container_volume)
            vessel.empty -= early_discharge_number
            port.empty += early_discharge_number
            vessel.early_discharge = early_discharge_number

    def _on_arrival(self, event: AtomEvent):
        """Handler for processing event when there is a vessel arriving at the port.

        When the vessel arriving at the port:
        1. Update the location index.
        2. Update the future stops information of this vessel.
        3. Update the vessel plan.

        Args:
            event (AtomEvent): Arrival event object.
        """

        arrival_payload: VesselStatePayload = event.payload
        vessel_idx = arrival_payload.vessel_idx
        vessel = self._vessels[vessel_idx]

        # Update vessel location so that later logic will get correct value.
        vessel.last_loc_idx = vessel.next_loc_idx
        vessel.is_parking = 1
        stop: Stop = self._data_cntr.vessel_stops[vessel.idx, vessel.next_loc_idx]
        vessel.loc_port_idx = stop.port_idx

        # We should update the future stop list once the vessel arrives.
        future_stops = self._data_cntr.vessel_future_stops[vessel.idx, vessel.last_loc_idx, vessel.next_loc_idx]
        vessel.set_stop_list(None, future_stops)

        # Update vessel plans.
        for plan_port_idx, plan_tick in self._data_cntr.vessel_planned_stops[
            vessel_idx, vessel.route_idx, vessel.last_loc_idx
        ]:
            self._vessel_plans[vessel_idx, plan_port_idx] = plan_tick

    def _on_departure(self, event: AtomEvent):
        """Handler for processing event when there is a vessel leaving from port.

        When the vessel departing from port:
        1. Update location to next stop.
        2. Update the past stops information of this vessel.

        Args:
            event (AtomEvent): Departure event object.
        """

        departure_payload: VesselStatePayload = event.payload
        vessel_idx = departure_payload.vessel_idx
        vessel = self._vessels[vessel_idx]

        # As we have unfold all the route stop, we can just location ++.
        vessel.next_loc_idx += 1
        vessel.is_parking = 0
        vessel.loc_port_idx = -1

        # We should update the past stop list once the vessel departs.
        past_stops = self._data_cntr.vessel_past_stops[vessel.idx, vessel.last_loc_idx, vessel.next_loc_idx]
        vessel.set_stop_list(past_stops, None)

    def _on_discharge(self, event: CascadeEvent):
        """Handler for processing event the there are some full need to be discharged.


        1. Discharge specified qty of full from vessel into port.on_consignee.
        2. Generate a empty_return event by configured buffer time:
            a. If buffer time is 0, then insert into immediate_event_list to process it ASAP.
            b. Or insert into event buffer.

        Args:
            event (AtomEvent): Discharge event object.
        """
        discharge_payload: VesselDischargePayload = event.payload
        vessel_idx = discharge_payload.vessel_idx
        port_idx = discharge_payload.port_idx
        vessel = self._vessels[vessel_idx]
        port = self._ports[port_idx]
        discharge_qty: int = discharge_payload.quantity

        vessel.full -= discharge_qty
        port.on_consignee += discharge_qty

        self._full_on_vessels[vessel_idx, port_idx] -= discharge_qty

        buffer_ticks = self._data_cntr.empty_return_buffers[port.idx]
        payload = EmptyReturnPayload(port_idx=port.idx, quantity=discharge_qty)
        mt_return_evt = self._event_buffer.gen_atom_event(
            tick=event.tick + buffer_ticks, event_type=Events.RETURN_EMPTY, payload=payload
        )

        if buffer_ticks == 0:
            event.add_immediate_event(mt_return_evt)
        else:
            self._event_buffer.insert_event(mt_return_evt)

    def _on_empty_return(self, event: AtomEvent):
        """Handler for processing event when there are some empty container return to port.

        Args:
            event (AtomEvent): Empty-return event object.
        """
        payload: EmptyReturnPayload = event.payload
        port = self._ports[payload.port_idx]

        port.on_consignee -= payload.quantity
        port.empty += payload.quantity

    def _on_action_received(self, event: CascadeEvent):
        """Handler for processing actions from agent.

        Args:
            event (CascadeEvent): Action event object with expected payload: {vessel_id: empty_number_to_move}}.
        """
        actions = event.payload

        if actions:
            if type(actions) is not list:
                actions = [actions]

            for action in actions:
                vessel_idx = action.vessel_idx
                port_idx = action.port_idx
                move_num = action.quantity
                vessel = self._vessels[vessel_idx]
                port = self._ports[port_idx]
                port_empty = port.empty
                vessel_empty = vessel.empty

                action_type: ActionType = getattr(action, "action_type", None)

                # Make it compatiable with previous action.
                if action_type is None:
                    action_type = ActionType.DISCHARGE if move_num > 0 else ActionType.LOAD

                # Make sure the move number is positive, as we have the action type.
                move_num = abs(move_num)

                if action_type == ActionType.DISCHARGE:
                    assert(move_num <= vessel_empty)

                    port.empty = port_empty + move_num
                    vessel.empty = vessel_empty - move_num
                else:
                    assert(move_num <= min(port_empty, vessel.remaining_space))

                    port.empty = port_empty - move_num
                    vessel.empty = vessel_empty + move_num

                # Align the event type to make the output readable.
                event.event_type = Events.DISCHARGE_EMPTY if action_type == ActionType.DISCHARGE else Events.LOAD_EMPTY

                # Update transfer cost for port and metrics.
                self._total_operate_num += move_num
                port.transfer_cost += move_num

                self._vessel_plans[vessel_idx, port_idx] += self._data_cntr.vessel_period[vessel_idx]

    def _stream_base_info(self):
        if streamit:
            streamit.info(self._scenario_name, self._topology, self._max_tick)
            streamit.complex("config", self._config)

    def _stream_data(self):
        if streamit:
            port_number = len(self._ports)
            vessel_number = len(self._vessels)

            for port in self._ports:
                streamit.data(
                    "port_details", index=port.index, capacity=port.capacity, empty=port.empty, full=port.full,
                    on_shipper=port.on_shipper, on_consignee=port.on_consignee, shortage=port.shortage,
                    acc_shortage=port.acc_shortage, booking=port.booking, acc_booking=port.acc_booking,
                    fulfillment=port.fulfillment, acc_fulfillment=port.acc_fulfillment, transfer_cost=port.transfer_cost
                )

            for vessel in self._vessels:
                streamit.data(
                    "vessel_details", index=vessel.index, capacity=vessel.capacity, empty=vessel.empty,
                    full=vessel.full, remaining_space=vessel.remaining_space, early_discharge=vessel.early_discharge,
                    route_idx=vessel.route_idx, last_loc_idx=vessel.last_loc_idx, next_loc_idx=vessel.next_loc_idx,
                    past_stop_list=vessel.past_stop_list[:], past_stop_tick_list=vessel.past_stop_tick_list[:],
                    future_stop_list=vessel.future_stop_list[:], future_stop_tick_list=vessel.future_stop_tick_list[:]
                )

            vessel_plans = np.array(self._vessel_plans[:]).reshape(vessel_number, port_number)

            a, b = np.where(vessel_plans > -1)

            for vessel_index, port_index in list(zip(a, b)):
                streamit.data(
                    "vessel_plans", vessel_index=vessel_index,
                    port_index=port_index, planed_arrival_tick=vessel_plans[vessel_index, port_index]
                )

            full_on_ports = np.array(self._full_on_ports[:]).reshape(port_number, port_number)

            a, b = np.where(full_on_ports > 0)

            for from_port_index, to_port_index in list(zip(a, b)):
                streamit.data(
                    "full_on_ports", from_port_index=from_port_index,
                    dest_port_index=to_port_index, quantity=full_on_ports[from_port_index, to_port_index]
                )

            full_on_vessels = np.array(self._full_on_vessels[:]).reshape(vessel_number, port_number)

            a, b = np.where(full_on_vessels > 0)

            for vessel_index, port_index in list(zip(a, b)):
                streamit.data(
                    "full_on_vessels", vessel_index=vessel_index, port_index=port_index,
                    quantity=full_on_vessels[vessel_index, port_index]
                )
