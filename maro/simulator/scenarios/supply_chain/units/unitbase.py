# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import List, Optional, Union

if typing.TYPE_CHECKING:
    from maro.simulator.scenarios.supply_chain.actions import SupplyChainAction
    from maro.simulator.scenarios.supply_chain.datamodels.base import DataModelBase
    from maro.simulator.scenarios.supply_chain.facilities import FacilityBase
    from maro.simulator.scenarios.supply_chain.world import World


@dataclass
class BaseUnitInfo:
    id: int
    node_index: int
    node_name: str
    class_name: type
    config: dict
    children: List[BaseUnitInfo]


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
    facility: Optional[FacilityBase] = None

    # Which world this unit belongs to.
    world: Optional[World] = None

    # Parent of this unit, it can be a facility or another unit.
    parent: Optional[Union[FacilityBase, UnitBase]] = None

    # Child units, extended unit can add their own child as property, this is used as a collection.
    children: Optional[list] = None

    # Data model name in the frame, used to query binding data model instance.
    data_model_name: Optional[str] = None

    # Data model instance index in the frame, used to query binding data model instance.
    data_model_index: Optional[int] = None

    # Real data model binding with this unit.
    data_model: Optional[DataModelBase] = None

    # Current action.
    action: Optional[SupplyChainAction] = None

    # Current unit configurations.
    config: Optional[dict] = None

    def __init__(self) -> None:
        pass

    def parse_configs(self, config: dict) -> None:
        """Parse configurations from config.

        Args:
            config (dict): Configuration from parent or config file.
        """
        self.config = config

    def step(self, tick: int) -> None:
        """Run related logic for current tick.

        Args:
            tick (int): Current simulator tick.
        """
        self._step_impl(tick)
        self._clear_action()

    def _step_impl(self, tick: int) -> None:
        pass

    def flush_states(self) -> None:
        """Flush states into frame for current tick.
        """
        pass

    def post_step(self, tick: int) -> None:
        """Post-processing for current step.

        Args:
            tick (int): Current simulator tick.
        """
        pass

    def reset(self) -> None:
        """Reset this unit for a new episode."""
        if self.data_model is not None:
            self.data_model.reset()

        self._clear_action()

    def initialize(self) -> None:
        """Initialize this unit after data model is ready to use.

        NOTE: unit.data_model is available from this step.
        """
        if self.data_model is not None:
            self.data_model.set_id(self.id, self.facility.id)

    def set_action(self, action: SupplyChainAction) -> None:
        """Set action for this agent.

        Args:
            action (object): Action from outside.
        """
        self.action = action

    def _clear_action(self) -> None:
        """Clear the action after calling step() of this Unit."""
        self.action = None

    def get_unit_info(self) -> BaseUnitInfo:
        return BaseUnitInfo(
            id=self.id,
            node_index=self.data_model_index,
            node_name=type(self.data_model).__node_name__,
            class_name=type(self),
            config=self.config,
            children=[c.get_unit_info() for c in self.children] if self.children else None,
        )
