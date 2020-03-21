# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from enum import IntEnum
from maro.simulator.frame import SnapshotList


class VesselState(IntEnum):
    """
    State of vessel
    """
    PARKING = 0
    SAILING = 1


class EcrEventType(IntEnum):
    """
    Event type for ECR problem
    """
    RELEASE_EMPTY = 10
    RETURN_FULL = 11
    LOAD_FULL = 12
    DISCHARGE_FULL = 13
    RELEASE_FULL = 14
    RETURN_EMPTY = 15
    ORDER = 16
    VESSEL_ARRIVAL = 17
    VESSEL_DEPARTURE = 18
    PENDING_DECISION = 19
    LOAD_EMPTY = 20
    DISCHARGE_EMPTY = 21


class Stop:
    """
    Present a stop in the vessel proforma, for internal using only
    """

    def __init__(self, arrive_tick: int, leave_tick: int, port_idx: int):
        """
        Create a new instance of Stop

        Args:
            arrive_tick (int): Tick that the vessel will arrive at port
            leave_tick (int): Tick that the vessel will leave the port
            port_idx (int): Port id that the vessel will arrive/leave
        """
        self.arrive_tick = arrive_tick
        self.leave_tick = leave_tick
        self.port_idx = port_idx

    def __repr__(self):
        return f"Stop {{arrive_tick:{self.arrive_tick}, leave_tick: {self.leave_tick}, port_id: {self.port_idx}}}"


# order object used to generate full
class Order:
    """
    Used to hold order information, this is for order generation
    """

    def __init__(self, tick: int, src_port_idx: int, dest_port_idx: int, quantity: int):
        """
        Create a new instants of order

        Args:
            tick (int): Generated tick of current order
            src_port_idx (int): Source port of this order
            dest_port_idx (int): Destination port id of this order
            quantity (int): Container quantity of this order
        """
        self.tick = tick
        self.src_port_idx = src_port_idx
        self.quantity = quantity
        self.dest_port_idx = dest_port_idx

    def __repr__(self):
        return f"Order {{tick:{self.tick}, source port: {self.src_port_idx}, dest port: {self.dest_port_idx} quantity: {self.quantity}}}"


# used for arrival and departure cascade event
class VesselStatePayload:
    """
    Payload object used to hold vessel state changes for event
    """

    def __init__(self, port_idx: int, vessel_idx: int):
        """

        Args:
            port_idx (int): Which port the vessel at
            vessel_idx (int): Which vessel's state changed
        """
        self.port_idx = port_idx
        self.vessel_idx = vessel_idx

    def __repr__(self):
        return f"VesselStatePayload {{ port: {self.port_idx}, vessel: {self.vessel_idx} }}"


class VesselDischargePayload:
    """
    Payload object to hold information about container discharge
    """

    def __init__(self, vessel_idx: int, from_port_idx: int, port_idx: int, quantity: int):
        """
        Create a new instance of VesselDischargePayload

        Args:
            vessel_idx (int): Which vessel will discharge
            from_port_idx (int): Which port sent the discharged containers
            port_idx (int): Which port will receive the discharged containers
            quantity (int): How many containers will be discharged

        """
        self.vessel_idx = vessel_idx
        self.from_port_idx = from_port_idx
        self.port_idx = port_idx
        self.quantity = quantity

    def __repr__(self):
        return f"VesselDischargePayload {{ vessel: {self.vessel_idx}, port: {self.port_idx}, qty: {self.quantity} }}"


class Action:
    """
    Action object
    """

    def __init__(self, vessel_idx: int, port_idx: int, quantity: int):
        """
        Create a new instance of VesselDischargePayload

        Args:
            vessel_idx (int): Which vessel will take action
            port_idx (int): Which port will take action
            quantity (int): How many containers can be moved from vessel to port (negative in reverse)

        """
        self.vessel_idx = vessel_idx
        self.port_idx = port_idx
        self.quantity = quantity

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'Action {{quantity: {self.quantity}, port: {self.port_idx}, vessel: {self.vessel_idx} }}'


class ActionScope:
    """
    Load and discharge scope for agent to generate decision
    """
    def __init__(self, load: int, discharge: int):
        self.load = load
        self.discharge = discharge

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'ActionScope {{load: {self.load}, discharge: {self.discharge} }}'


class DecisionEvent:
    """
    Decision event for agent
    """

    def __init__(self, tick: int, port_idx: int, vessel_idx: int, snapshot_list: SnapshotList,
                 action_scope_func, early_discharge_func):
        """
        Create a new instance of CascadeEventPayload

        Args:
            tick (int): On which tick we need an action
            port_idx (int): Which port will take action
            vessel_idx (int): Which vessel will take action
            snapshot_list (int): Snapshots of the environment to input into the decision model
            action_scope_func (Function): Function to calculate action scope, we use function here to make it
                                            to get the value as late as possible
            early_discharge_func (Function): Function to fetch early discharge number of spedified vessel, we
                                            use function here to make it to get the value as late as possible
        """
        self.tick = tick
        self.port_idx = port_idx
        self.vessel_idx = vessel_idx
        self.snapshot_list = snapshot_list
        self._action_scope = None  # this field will be fixed after the action_scope property is called 1st time
        self._early_discharge = None  # this field will be fixed after the early_discharge property is called 1st time
        self._action_scope_func = action_scope_func
        self._early_discharge_func = early_discharge_func

    @property
    def action_scope(self) -> ActionScope:
        """
        Load and discharge scope for agent to generate decision
        """
        if self._action_scope is None:
            self._action_scope = self._action_scope_func(self.port_idx, self.vessel_idx)

        return self._action_scope

    @property
    def early_discharge(self) -> int:
        """
        Early discharge number of corresponding vessel
        """
        if self._early_discharge is None:
            self._early_discharge = self._early_discharge_func(self.vessel_idx)

        return self._early_discharge

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'DecisionEvent(tick={self.tick}, port_idx={self.port_idx}, vessel_idx={self.vessel_idx}, action_scope={self.action_scope})'
