# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from enum import Enum, IntEnum

from maro.backends.frame import SnapshotList


class VesselState(IntEnum):
    """State of vessel."""
    PARKING = 0
    SAILING = 1


class ActionType(Enum):
    """Type of CIM action."""
    LOAD = "load",
    DISCHARGE = "discharge"


class Action:
    """Action object that used to pass action from agent to business engine.

    Args:
        vessel_idx (int): Which vessel will take action.
        port_idx (int): Which port will take action.
        action_type (ActionType): Whether the action is a Load or a Discharge.
        quantity (int): How many containers are loaded/discharged in this Action.
    """

    summary_key = ["port_idx", "vessel_idx", "action_type", "quantity"]

    def __init__(self, vessel_idx: int, port_idx: int, quantity: int, action_type: ActionType):
        self.vessel_idx = vessel_idx
        self.port_idx = port_idx
        self.quantity = quantity
        self.action_type = action_type

    def __repr__(self):
        return "%s {action_type: %r, port_idx: %r, vessel_idx: %r, quantity: %r}" % \
            (self.__class__.__name__, str(self.action_type), self.port_idx, self.vessel_idx, self.quantity)


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
        return "%s {load: %r, discharge: %r}" % \
            (self.__class__.__name__, self.load, self.discharge)


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
    summary_key = ["tick", "port_idx", "vessel_idx", "snapshot_list", "action_scope", "early_discharge"]

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

        return int(self._early_discharge)

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

    def __setstate__(self, state):
        self.tick = state["tick"]
        self.port_idx = state["port_idx"]
        self.vessel_idx = state["vessel_idx"]
        self._action_scope = state["action_scope"]
        self._early_discharge = state["early_discharge"]

    def __repr__(self):
        return "%s {port_idx: %r, vessel_idx: %r, action_scope: %r, early_discharge: %r}" % \
            (self.__class__.__name__, self.port_idx, self.vessel_idx, self.action_scope, self.early_discharge)
