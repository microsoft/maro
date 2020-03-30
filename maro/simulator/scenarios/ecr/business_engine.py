# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import os
from math import ceil, floor

from yaml import safe_load

from maro.simulator.event_buffer import DECISION_EVENT, Event, EventBuffer
from maro.simulator.frame import Frame, SnapshotList
from maro.simulator.scenarios import AbsBusinessEngine

from .common import (ActionScope, DecisionEvent, EcrEventType, Order, Stop,
                     VesselDischargePayload, VesselStatePayload)
from .ecr_data_generator import EcrDataGenerator
from .frame_builder import gen_ecr_frame
from .matrix_accessor import FrameMatrixAccessor
from .port import Port
from .vessel import Vessel


class EcrBusinessEngine(AbsBusinessEngine):
    def __init__(self, event_buffer: EventBuffer, config_path: str, start_tick: int, max_tick: int, frame_resolution: int):
        """
        Create a new instance of ECR Business Engine

        Args:
            event_buffer (EventBuffer): used to register and hold events
            topology_path (str): full path to the topology folder
            max_tick (int): max tick that we will simulate
        """
        super().__init__(event_buffer, config_path, start_tick, max_tick, frame_resolution)        

        config_path = os.path.join(config_path, "config.yml")

        self._data_generator = EcrDataGenerator(max_tick, config_path)

        # create a copy of config object to expose to others, and not affect generator
        with open(config_path) as fp:
            self._config = safe_load(fp)

        self._vessels = []
        self._ports = []
        self._frame = None
        self._full_on_ports: FrameMatrixAccessor = None
        self._full_on_vessels: FrameMatrixAccessor = None
        self._vessel_plans: FrameMatrixAccessor = None

        self._init_frame()

        # snapshot list should be initialized after frame
        self._snapshots = SnapshotList(self._frame, max_tick)

        self._register_events()

        # as we already unpack the route to the max tick, so we can insert all departure event at the beginning
        self._load_departure_events()

    @property
    def configs(self):
        """
        Configurations of ECR business engine
        """
        return self._data_generator.get_pure_config()

    @property
    def frame(self) -> Frame:
        """
        Frame of current business engine
        """
        return self._frame

    @property
    def snapshots(self) -> SnapshotList:
        """
        Snapshot list of current frame
        """
        return self._snapshots

    def step(self, tick: int):
        """
        Called at each tick to generate orders and arrival events

        Args:
            tick (int): Tick to generate orders
        """

        # At each tick:
        # 1. generate orders for this tick
        # 2. transfer orders into events (ORDER)
        # 3. check and add vessel arrive event (atom and cascade)

        total_empty_number = sum([node.empty for node in self._ports + self._vessels])

        for order in self._data_generator.generate_orders(tick, total_empty_number):
            order_evt = self._event_buffer.gen_atom_event(tick, EcrEventType.ORDER, order)

            self._event_buffer.insert_event(order_evt)

        # used to hold cascade event of this tick, we will append this at the end
        # to make sure all the other logic finished
        # TODO: this can be remove after we support event priority
        cascade_evt_list = []

        for vessel in self._vessels:
            vessel_idx = vessel.idx
            loc_idx = vessel.next_loc_idx
            stop: Stop = self._data_generator.get_stop_from_idx(vessel_idx, loc_idx)
            port_idx = stop.port_idx

            # at the beginning the vessel is parking at port, will not invoke arrive event
            if loc_idx > 0:

                # check if there is any arrive event
                if stop.arrive_tick == tick:
                    arrival_payload = VesselStatePayload(port_idx, vessel_idx)

                    # this vessel will arrive at current tick
                    arrival_event = self._event_buffer.gen_atom_event(tick, EcrEventType.VESSEL_ARRIVAL,arrival_payload)

                    # then it will load full
                    load_event = self._event_buffer.gen_atom_event(tick, EcrEventType.LOAD_FULL, arrival_payload)

                    self._event_buffer.insert_event(arrival_event)
                    self._event_buffer.insert_event(load_event)

                    # generate cascade event and payload
                    decision_payload = DecisionEvent(tick, port_idx, vessel_idx, self.snapshots,
                                                     self.action_scope, self.early_discharge)

                    decision_event = self._event_buffer.gen_cascade_event(tick,
                                                                          EcrEventType.PENDING_DECISION,
                                                                          decision_payload)

                    cascade_evt_list.append(decision_event)

                    # update vessel location so that later logic will get correct value
                    vessel.last_loc_idx = vessel.next_loc_idx

            # we should update the future stop list at each tick
            vessel.set_stop_list(self._data_generator.get_stop_list(vessel.idx, vessel.last_loc_idx, loc_idx))

            # update vessel plans
            for plan_port_idx, plan_tick in self._data_generator.get_planed_stops(vessel_idx, vessel.route_idx,
                                                                                  loc_idx):
                self._vessel_plans[vessel_idx: plan_port_idx] = plan_tick

            if loc_idx > 0 and stop.arrive_tick == tick:
                self._vessel_plans[vessel_idx: port_idx] = stop.arrive_tick

        # insert the cascade events at the end
        for evt in cascade_evt_list:
            self._event_buffer.insert_event(evt)

        return tick + 1 == self._max_tick

    def post_step(self, tick: int):
        """
        Post-process after each step

        Args:
            tick (int): tick to process
        """
        # update acc_fulfillment before take snapshot
        for port in self._ports:
            port.acc_fulfillment = port.acc_booking - port.acc_shortage

        # before go to next tick, we will take a snapshot first
        self._snapshots.insert_snapshot(self._frame, tick)

        # reset port statistics (by tick) fields
        for port in self._ports:
            port.shortage = 0
            port.booking = 0
            port.fulfillment = 0

    def rewards(self, actions: list):
        """
        Reward base on actions

        Args:
            Actions list(action): Action list from agent: {vessel_id: empty_number_to_move}

        Returns:
            Corresponding reward list
        """
        if actions is None:
            return []

        self_rewards = [(port.booking - port.shortage) * 0.05 for port in self._ports]
        average_reward = sum(self_rewards) / self._data_generator.port_num
        rewards = [self_reward * 0.5 + average_reward * 0.5 for self_reward in self_rewards]

        return [rewards[action.port_idx] for action in actions]

    def reset(self):
        """
        Reset the business engine, it will reset frame value
        """

        self._snapshots.reset()

        self._frame.reset()

        self._reset_nodes()

        # insert departure event again
        self._load_departure_events()

    def action_scope(self, port_idx: int, vessel_idx: int) -> ActionScope:
        """
        Get the action scope of specified agent

        Args:
            port_idx (int): Index of specified agent
            vessel_idx (int): Index of specified vessel to take the action

        Returns:
            Scope dictionary {
                "port2vessel": Max number of empty containers that can be moved from port to vessel,
                "vessel2port": Max number of empty containers that can be moved from vessel to port
            }
        """
        port: Port = self._ports[port_idx]
        vessel: Vessel = self._vessels[vessel_idx]
        vessel_empty = vessel.empty
        vessel_total_space = int(floor(vessel.capacity / self._data_generator.container_volume))
        vessel_remaining_space = vessel_total_space - vessel.full - vessel_empty
        vessel.remaining_space = vessel_remaining_space

        return ActionScope(load=min(port.empty, vessel_remaining_space), discharge=vessel_empty)

    def early_discharge(self, vessel_idx: int) -> int:
        """
        Get the early discharge number of specified vessel

        Args:
            vessel_idx (int): Index of specified vessel
        """
        return self._vessels[vessel_idx].early_discharge

    def get_node_name_mapping(self) -> dict:
        """
        Get node name mappings related with this environment

        Returns:
            Node name to index mapping dictionary
            {
                "static": {name: index}
                "dynamic": {name: index}
            }
        """
        return self._data_generator.node_mapping

    def get_agent_idx_list(self) -> list:
        """
        Get port index list related with this environment

        Returns: 
            A list of port index
        """
        return [i for i in range(self._data_generator.port_num)]

    def _reset_nodes(self):
        for port in self._ports:
            port.reset()
            port.capacity = self._data_generator.port_initial_info["capacity"][port.idx]
            port.empty = self._data_generator.port_initial_info["empty"][port.idx]

        for vessel in self._vessels:
            vessel.reset()
            vessel.capacity = self._data_generator.vessel_initial_info["capacity"][vessel.idx]
            vessel.remaining_space = int(vessel.capacity / self._data_generator.container_volume)
            vessel.route_idx = self._data_generator.vessel_initial_info["route"][vessel.idx]
            vessel.early_discharge = 0

            if "empty" in self._data_generator.vessel_initial_info:
                vessel.empty = self._data_generator.vessel_initial_info["empty"][vessel.idx]

            vessel.next_loc_idx = 0
            vessel.last_loc_idx = 0

        # reset vessel plans
        for vessel_idx in range(self._data_generator.vessel_num):
            for port_idx in range(self._data_generator.port_num):
                self._vessel_plans[vessel_idx: port_idx] = -1

    def _register_events(self):
        """
        Register events
        """
        register_handler = self._event_buffer.register_event_handler

        register_handler(EcrEventType.RETURN_FULL, self._on_full_return)
        register_handler(EcrEventType.RETURN_EMPTY, self._on_empty_return)
        register_handler(EcrEventType.ORDER, self._on_order_generated)
        register_handler(EcrEventType.LOAD_FULL, self._on_full_load)
        register_handler(EcrEventType.VESSEL_DEPARTURE, self._on_departure)
        register_handler(EcrEventType.DISCHARGE_FULL, self._on_discharge)
        register_handler(DECISION_EVENT, self._on_action_received)

    def _load_departure_events(self):
        """
        Insert leaving event at the beginning as we already unpack the root to a loop at the beginning
        """

        for vessel_idx, stops in enumerate(self._data_generator.vessel_stops):
            for stop in stops:
                payload = VesselStatePayload(stop.port_idx, vessel_idx)
                dep_evt = self._event_buffer.gen_atom_event(stop.leave_tick, EcrEventType.VESSEL_DEPARTURE, payload)
                self._event_buffer.insert_event(dep_evt)

    def _init_frame(self):
        """
        Initialize the frame basing on data generator
        """
        self._frame = gen_ecr_frame(self._data_generator.port_num,
                                    self._data_generator.vessel_num,
                                    self._data_generator.stop_number)

        for port_idx, port_name in self._data_generator.node_mapping["static"].items():
            self._ports.append(Port(self._frame, port_idx, port_name))

        for vessel_idx, vessel_name in self._data_generator.node_mapping["dynamic"].items():
            self._vessels.append(Vessel(self._frame, vessel_idx, vessel_name))

        self._full_on_ports = FrameMatrixAccessor(self._frame, "full_on_ports")
        self._full_on_vessels = FrameMatrixAccessor(self._frame, "full_on_vessels")
        self._vessel_plans = FrameMatrixAccessor(self._frame, "vessel_plans")

        self._reset_nodes()

    def _get_reachable_ports(self, vessel_idx: int):
        """
        Get ports that specified vessel can reach (for order), return a list of tuple (port_id, arrive_tick)

        Args:
            vessel_idx (int): Index of specified vessel

        Returns:
            Reachable port index list of specified vessel
        """
        vessel = self._vessels[vessel_idx]

        return self._data_generator.get_reachable_stops(vessel_idx, vessel.route_idx, vessel.next_loc_idx)

    def _get_pending_full(self, src_port_idx: int, dest_port_idx: int):
        """
        Get pending full number from src_port_idx to dest_port_idx
        """
        return self._full_on_ports[src_port_idx: dest_port_idx]

    def _set_pending_full(self, src_port_idx: int, dest_port_idx: int, value):
        """
        Set the full number from src_port_idx to dest_port_idx
        """
        assert value >= 0

        self._full_on_ports[src_port_idx: dest_port_idx] = value

    def _on_order_generated(self, evt: Event):
        """
        When there is an order generated, we should do:
        1. Generate a LADEN_RETURN event by configured buffer time
           a. If the configured buffer time is 0, then insert the event to immediate_event_list to process it ASAP
           b. Or insert the event to event buffer directly

        2. Update port state: on_shipper +, empty -

        Args:
            evt (Event Object): Order event object
        """
        order: Order = evt.payload
        src_port: Port = self._ports[order.src_port_idx]

        execute_qty = order.quantity
        src_empty = src_port.empty
        src_port.booking += execute_qty
        src_port.acc_booking += execute_qty

        # check if there is any shortage
        if src_empty < order.quantity:
            # booking & shortage
            shortage_qty = order.quantity - src_empty
            src_port.shortage += shortage_qty
            src_port.acc_shortage += shortage_qty
            execute_qty = src_empty

        # update port state
        src_port.empty -= execute_qty
        src_port.on_shipper += execute_qty  # full that pending return

        buffer_ticks = self._data_generator.get_full_buffer_tick(src_port.idx)

        payload = (order.src_port_idx, order.dest_port_idx, execute_qty)
        laden_return_evt = self._event_buffer.gen_atom_event(evt.tick + buffer_ticks, EcrEventType.RETURN_FULL, payload)

        # if buffer_tick is 0, we should execute it as this tick
        if buffer_ticks == 0:
            evt.immediate_event_list.append(laden_return_evt)
        else:
            self._event_buffer.insert_event(laden_return_evt)

    def _on_full_return(self, evt: Event):
        """
        The full is ready to be loaded now.
        1. move it from on_shipper to full (update state: on_shipper -> full),
           and append it to the port pending list

        """
        # the payload is a tuple (src_port_idx, dest_port_idx, quantity)
        full_rtn_payload = evt.payload
        src_port_idx = full_rtn_payload[0]
        dest_port_idx = full_rtn_payload[1]
        qty = full_rtn_payload[2]

        src_port: Port = self._ports[src_port_idx]
        src_port.on_shipper -= qty
        src_port.full += qty

        pending_full_number = self._get_pending_full(src_port_idx, dest_port_idx)

        self._set_pending_full(src_port_idx, dest_port_idx, pending_full_number + qty)

    def _on_full_load(self, evt: Event):
        """
        Handler to process event that a vessel need to load full containers from current port.

        When there is a vessel arrive at a port:
        1. Discharge full (we ignore this action here, as we will generate a discharge event after
           a vessel have loaded any full)
        2. Load full by destination id, and generate discharge event
        3. Update vessel.state to PARKING
        4. Fill future stop list
        5. Early discharge

        Args:
            evt (Event Object): Arrival event object
        """

        arrival_obj: VesselStatePayload = evt.payload
        vessel_idx: int = arrival_obj.vessel_idx
        port_idx: int = arrival_obj.port_idx
        vessel: Vessel = self._vessels[vessel_idx]
        port: Port = self._ports[port_idx]
        container_volume = self._data_generator.container_volume
        vessel_capacity = vessel.capacity

        # update vessel state
        vessel.last_loc_idx = vessel.next_loc_idx

        remaining_space = vessel_capacity - vessel.full * container_volume

        # how many containers we can load
        acceptable_number = floor(remaining_space / container_volume)

        total_load_qty = 0

        for next_port_idx, arrive_tick in self._get_reachable_ports(vessel_idx):
            full_number_to_next_port = self._get_pending_full(port_idx, next_port_idx)

            if acceptable_number > 0 and full_number_to_next_port > 0:
                # we can load some full
                loaded_qty = min(full_number_to_next_port, acceptable_number)
                total_load_qty += loaded_qty

                # update port state
                self._set_pending_full(port_idx, next_port_idx, full_number_to_next_port - loaded_qty)

                port.full -= loaded_qty
                vessel.full += loaded_qty

                # update state
                self._full_on_vessels[vessel_idx: next_port_idx] += loaded_qty

                acceptable_number -= loaded_qty

                # generate a discharge event, as we know when the vessel will arrive at destination
                payload = VesselDischargePayload(vessel_idx, port_idx, next_port_idx, loaded_qty)
                dsch_event = self._event_buffer.gen_atom_event(arrive_tick, EcrEventType.DISCHARGE_FULL, payload)

                self._event_buffer.insert_event(dsch_event)

        # early discharge
        total_container = vessel.full + vessel.empty
        container_volume = self._data_generator.container_volume

        vessel.early_discharge = 0
        if total_container * container_volume > vessel.capacity:
            early_discharge_number = total_container - ceil(vessel.capacity / container_volume)
            vessel.empty -= early_discharge_number
            port.empty += early_discharge_number
            vessel.early_discharge = early_discharge_number

        # update remaining space
        vessel.remaining_space = vessel.capacity - (vessel.empty + vessel.full) * container_volume

    def _on_departure(self, evt: Event):
        """
        Handler to process event when there is a vessel leaving from port

        When the vessel departing from port:
        1. Update location to next stop

        Args:
            evt (Event Object): Departure event object
        """

        departure_payload: VesselStatePayload = evt.payload
        vessel_idx = departure_payload.vessel_idx
        vessel: Vessel = self._vessels[vessel_idx]

        # as we have unfold all the route stop, we can just location ++
        vessel.next_loc_idx += 1

    def _on_discharge(self, evt: Event):
        """
        Handler to process event the there are some full need to be discharged


        1. Discharge specified qty of full from vessel into port.on_consignee
        2. Generate a mt_return event by configured buffer time
            a. If buffer time is 0, then insert into immediate_event_list to process it ASAP
            b. Or insert into event buffer

        Args:
            evt (Event Object): Discharge event object
        """
        discharge_payload: VesselDischargePayload = evt.payload
        vessel_idx = discharge_payload.vessel_idx
        port_idx = discharge_payload.port_idx
        vessel: Vessel = self._vessels[vessel_idx]
        port: Port = self._ports[port_idx]
        discharge_qty: int = discharge_payload.quantity

        vessel.full -= discharge_qty
        port.on_consignee += discharge_qty
        vessel.remaining_space = vessel.capacity - (vessel.empty + vessel.full) * self._data_generator.container_volume

        self._full_on_vessels[vessel_idx: port_idx] -= discharge_qty

        buffer_ticks = self._data_generator.get_empty_buffer_tick(port.idx)
        payload = (discharge_qty, port.idx)
        mt_return_evt = self._event_buffer.gen_atom_event(evt.tick + buffer_ticks, EcrEventType.RETURN_EMPTY, payload)

        if buffer_ticks == 0:
            evt.immediate_event_list.append(mt_return_evt)
        else:
            self._event_buffer.insert_event(mt_return_evt)

    def _on_empty_return(self, evt: Event):
        """
        Handler to process event when there are some empty container return to port

        Args:
            evt (Event Object): Empty-return event object
        """
        qty, port_idx = evt.payload
        port: Port = self._ports[port_idx]

        port.on_consignee -= qty
        port.empty += qty

    def _on_action_received(self, evt: Event):
        """
        Handler to process actions from agent

        Args:
            evt (Event Object): Action event object with expected payload: {vessel_id: empty_number_to_move}}
        """
        actions = evt.payload

        if actions:
            if type(actions) is not list:
                actions = [actions]

            container_volume = self._data_generator.container_volume

            for action in actions:
                vessel_idx = action.vessel_idx
                port_idx = action.port_idx
                move_num = action.quantity
                vessel = self._vessels[vessel_idx]
                port = self._ports[port_idx]
                port_empty = port.empty
                vessel_empty = vessel.empty
                vessel_full = vessel.full

                vessel_total_space = int(floor(vessel.capacity / container_volume))
                vessel_remaining_space = vessel_total_space - vessel_empty - vessel_full

                assert -min(port.empty, vessel_remaining_space) <= move_num <= vessel_empty

                port.empty = port_empty + move_num
                vessel.empty = vessel_empty - move_num
                vessel.remaining_space = (vessel_total_space - vessel.empty - vessel_full) * container_volume

                evt.event_type = EcrEventType.DISCHARGE_EMPTY if move_num > 0 else EcrEventType.LOAD_EMPTY

                self._vessel_plans[vessel_idx: port_idx] += self._data_generator.get_vessel_period(vessel_idx)
