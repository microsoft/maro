# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


class UnitBase:
    """Base of all unit used to contain related logic.

    Typically one unit instance should bind to a data model instance,
    that used to update states in frame.

    An unit will have following steps to initializing.
    . Create instance with default constructor without parameter, means all unit constructor should not have parameter
        or with default value.
    . Unit.parse_configs is called with configurations from parent or config file.
    . After frame is constructed, Unit.initialize is called to do data model related initializing.
    . At the beginning of business_engine.step, Unit.step is called to go through related logic,
        then after all the agent completed, Unit.flush_state will be called at the end of business_engine.step,
        to tell agents to save their states.
    . Unit.post_step is called in business_engine.post_step after step is finished.
    . Unit.set_action is called when there is any action from out-side.

    """
    # Id of this unit.
    id: int = 0

    # Which this unit belongs to.
    facility = None

    # Which world this unit belongs to.
    world = None

    # Parent of this unit, it can be a facility or another unit.
    parent: object = None

    # Child units, extended unit can add their own child as property, this is used as a collection.
    children: list = None

    # Data model name in the frame, used to query binding data model instance.
    data_model_name: str = None

    # Data model instance index in the frame, used to query binding data model instance.
    data_model_index: int = None

    # Real data model binding with this unit.
    data_model = None

    # Current action.
    action: object = None

    # Current unit configurations.
    config: dict = None

    def parse_configs(self, config: dict):
        """Parse configurations from config.

        Args:
            config (dict): Configuration from parent or config file.
        """
        self.config = config

    def step(self, tick: int):
        """Run related logic for current tick.

        Args:
            tick (int): Current simulator tick.
        """
        pass

    def flush_states(self):
        """Flush states into frame for current tick.
        """
        pass

    def post_step(self, tick: int):
        """Post-processing for current step.

        Args:
            tick (int): Current simulator tick.
        """
        self.action = None

    def reset(self):
        """Reset this unit for a new episode."""
        if self.data_model is not None:
            self.data_model.reset()

    def initialize(self):
        """Initialize this unit after data model is ready to use.

        NOTE: unit.data_model is available from this step.
        """
        if self.data_model is not None:
            self.data_model.set_id(self.id, self.facility.id)

    def set_action(self, action: object):
        """Set action for this agent.

        Args:
            action (object): Action from outside.
        """
        self.action = action

    def get_unit_info(self) -> dict:
        return {
            "id": self.id,
            "node_name": type(self.data_model).__node_name__,
            "node_index": self.data_model_index,
            "class": type(self)
        }
