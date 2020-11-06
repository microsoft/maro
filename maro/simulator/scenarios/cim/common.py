# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from enum import IntEnum

from maro.backends.frame import SnapshotList


class VesselState(IntEnum):
    """State of vessel.
    """
    PARKING = 0
    SAILING = 1


class CimEventType(IntEnum):
    """Event type for CIM problem.
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


# used for arrival and departure cascade event
class VesselStatePayload:
    """Payload object used to hold vessel state changes for event.

    Args:
        port_idx (int): Which port the vessel at.
        vessel_idx (int): Which vessel's state changed.
    """
    def __init__(self, port_idx: int, vessel_idx: int):

        self.port_idx = port_idx
        self.vessel_idx = vessel_idx

    def __repr__(self):
        return f"VesselStatePayload {{ port: {self.port_idx}, vessel: {self.vessel_idx} }}"


class VesselDischargePayload:
    """Payload object to hold information about container discharge.

    Args:
        vessel_idx (int): Which vessel will discharge.
        from_port_idx (int): Which port sent the discharged containers.
        port_idx (int): Which port will receive the discharged containers.
        quantity (int): How many containers will be discharged.
    """
    def __init__(self, vessel_idx: int, from_port_idx: int, port_idx: int, quantity: int):
        self.vessel_idx = vessel_idx
        self.from_port_idx = from_port_idx
        self.port_idx = port_idx
        self.quantity = quantity

    def __repr__(self):
        return f"VesselDischargePayload {{ vessel: {self.vessel_idx}, port: {self.port_idx}, qty: {self.quantity} }}"


class Action:
    """Action object that used to pass action from agent to business engine.

    Args:
        vessel_idx (int): Which vessel will take action.
        port_idx (int): Which port will take action.
        quantity (int): How many containers can be moved from vessel to port (negative in reverse).
    """

    def __init__(self, vessel_idx: int, port_idx: int, quantity: int):
        self.vessel_idx = vessel_idx
        self.port_idx = port_idx
        self.quantity = quantity

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'Action(port_idx={self.port_idx}, vessel_idx={self.vessel_idx}, quantity={self.quantity})'


class ActionScope:
    """Load and discharge scope for agent to generate decision.

    Args:
        load (int): Max number to load.
        discharge (int): Max number to discharge.
    """

    def __init__(self, load: int, discharge: int):
        self.load = load
        self.discharge = discharge

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'ActionScope(load={self.load}, discharge={self.discharge})'


class DecisionEvent:
    """Decision event for agent.

    Args:
        tick (int): On which tick we need an action.
        port_idx (int): Which port will take action.
        vessel_idx (int): Which vessel will take action.
        snapshot_list (int): Snapshots of the environment to input into the decision model.
        action_scope_func (Function): Function to calculate action scope, we use function here to make it
            getting the value as late as possible.
        early_discharge_func (Function): Function to fetch early discharge number of specified vessel, we
            use function here to make it getting the value as late as possible.
    """
    def __init__(
        self, tick: int, port_idx: int, vessel_idx: int, snapshot_list: SnapshotList,
        action_scope_func, early_discharge_func
    ):
        self.tick = tick
        self.port_idx = port_idx
        self.vessel_idx = vessel_idx
        self.snapshot_list = snapshot_list
        # this field will be fixed after the action_scope property is called 1st time
        self._action_scope = None
        # this field will be fixed after the early_discharge property is called 1st time
        self._early_discharge = None
        self._action_scope_func = action_scope_func
        self._early_discharge_func = early_discharge_func

    @property
    def action_scope(self) -> ActionScope:
        """ActionScope: Load and discharge scope for agent to generate decision.
        """
        if self._action_scope is None:
            self._action_scope = self._action_scope_func(self.port_idx, self.vessel_idx)

        return self._action_scope

    @property
    def early_discharge(self) -> int:
        """int: Early discharge number of corresponding vessel.
        """
        if self._early_discharge is None:
            self._early_discharge = self._early_discharge_func(self.vessel_idx)

        return self._early_discharge

    def __getstate__(self):
        """Return pickleable dictionary.

        NOTE: this class do not support unpickle"""
        return {
            "tick": self.tick,
            "port_idx": self.port_idx,
            "vessel_idx": self.vessel_idx,
            "action_scope": self.action_scope,
            "early_discharge": self.early_discharge
        }

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'DecisionEvent(tick={self.tick}, port_idx={self.port_idx}, \
            vessel_idx={self.vessel_idx}, action_scope={self.action_scope})'
