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
    node_name: Optional[str]
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
    def __init__(
        self, id: int, data_model_name: Optional[str], data_model_index: Optional[int],
        facility: FacilityBase, parent: Union[FacilityBase, UnitBase], world: World, config: dict,
    ) -> None:

        # Id of this unit.
        self.id: int = id

        # Data model name in the frame, used to query binding data model instance.
        self.data_model_name: Optional[str] = data_model_name
        # Data model instance index in the frame, used to query binding data model instance.
        self.data_model_index: Optional[int] = data_model_index

        # Which this unit belongs to.
        self.facility: FacilityBase = facility

        # Parent of this unit, it can be a facility or another unit.
        self.parent: Union[FacilityBase, UnitBase] = parent

        # Which world this unit belongs to.
        self.world: World = world

        # Current unit configurations.
        self.config: dict = config

        # Child units, extended unit can add their own child as property, this is used as a collection.
        self.children: Optional[list] = None

        # Real data model binding with this unit.
        self.data_model: Optional[DataModelBase] = None

    def pre_step(self, tick: int) -> None:
        pass

    def step(self, tick: int) -> None:
        """Run related logic for current tick.

        Args:
            tick (int): Current simulator tick.
        """
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

    def initialize(self) -> None:
        """Initialize this unit after data model is ready to use.

        NOTE: unit.data_model is available from this step.
        """
        if self.data_model is not None:
            self.data_model.set_id(self.id, self.facility.id)

    def on_action_received(self, tick: int, action: SupplyChainAction) -> None:
        """Action handler for each unit.

        Args:
            tick (int): Tick when action received.
            action (object): Action from outside.
        """
        pass

    def get_unit_info(self) -> BaseUnitInfo:
        return BaseUnitInfo(
            id=self.id,
            node_index=self.data_model_index,
            node_name=type(self.data_model).__node_name__ if self.data_model else None,
            class_name=type(self),
            config=self.config,
            children=[c.get_unit_info() for c in self.children] if self.children else None,
        )
