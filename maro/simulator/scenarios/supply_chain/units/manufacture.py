# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import typing
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, Union

from maro.simulator.scenarios.supply_chain.actions import ManufactureAction
from maro.simulator.scenarios.supply_chain.datamodels import ManufactureDataModel

from .extendunitbase import ExtendUnitBase, ExtendUnitInfo
from .unitbase import UnitBase

if typing.TYPE_CHECKING:
    from maro.simulator.scenarios.supply_chain.facilities import FacilityBase
    from maro.simulator.scenarios.supply_chain.world import World


@dataclass
class ManufactureUnitInfo(ExtendUnitInfo):
    pass


class ManufactureUnit(ExtendUnitBase):
    """Unit that used to produce certain product(sku) with consume specified source skus.

    One manufacture unit per sku.
    """
    def __init__(
        self, id: int, data_model_name: Optional[str], data_model_index: Optional[int],
        facility: FacilityBase, parent: Union[FacilityBase, UnitBase], world: World, config: dict,
    ) -> None:
        super(ManufactureUnit, self).__init__(id, data_model_name, data_model_index, facility, parent, world, config)

        # Source material sku and related quantity per manufacture cycle.
        self._bom: Optional[dict] = None
        # How many products in each manufacture cycle.
        self._output_units_per_lot: Optional[int] = None
        # How many units we will consume in each manufacture cycle.
        self._input_units_per_lot: int = 0
        self._space_taken_per_lot: int = 1

        self._unit_product_cost: Optional[float] = None
        self._max_manufacture_rate: Optional[int] = None
        self._manufacture_leading_time: Optional[int] = None

        self._manufacture_rate: Optional[int] = None
        self._num_to_produce: int = 0
        self._in_pipeline_quantity: int = 0
        self._finished_quantity: int = 0
        self._manufacture_cost: float = 0

        # Key: expected finished tick; Value: output product quantity.
        self._products_in_pipeline: Dict[int, int] = defaultdict(int)

    def initialize(self) -> None:
        super(ManufactureUnit, self).initialize()

        # Initialize BOM info.
        global_sku_info = self.world.get_sku_by_id(self.sku_id)

        self._bom = global_sku_info.bom
        self._output_units_per_lot = global_sku_info.output_units_per_lot

        if len(self._bom) > 0:
            self._input_units_per_lot = sum(self._bom.values())

        self._space_taken_per_lot = self._output_units_per_lot - self._input_units_per_lot

        # Initialize SKU info.
        self._unit_product_cost = self.facility.skus[self.sku_id].unit_product_cost
        self._max_manufacture_rate = self.facility.skus[self.sku_id].max_manufacture_rate
        self._manufacture_leading_time = self.facility.skus[self.sku_id].manufacture_leading_time
        assert self._unit_product_cost is not None
        assert self._max_manufacture_rate is not None
        assert self._manufacture_leading_time is not None

        self._manufacture_rate = self._max_manufacture_rate

        # Initialize data model.
        assert isinstance(self.data_model, ManufactureDataModel)
        self.data_model.initialize()

    """
    ManufactureAction would be given after BE.step(), assume we take action at t0,
    the manufacture_rate would be set according to the given action and would start manufacture in t0,
    then we can get the produced products at the end of (t0 + leading time) in the post_step(), which means
    these products can't be dispatched to fulfill the demand from the downstreams until (t0 + leading time + 1).
    """

    def process_action(self, tick: int, action: ManufactureAction) -> None:
        # NOTE: the process_action() is called after flush_state(), so the manufacture_rate saved in the snapshot
        # would be the one actually used to produce products in this tick.
        self._manufacture_rate = max(0, min(action.manufacture_rate, self._max_manufacture_rate))

    def step(self, tick: int) -> None:
        pass

    def _manufacture(self, tick: int) -> None:
        # Update num_to_produce according to limitations.
        self._num_to_produce = self._manufacture_rate * self._output_units_per_lot

        # Check the remaining space limits. TODO: confirm the remaining space setting.
        if self._num_to_produce > 0:
            remaining_space = self.facility.storage.get_product_max_remaining_space(self.sku_id)
            self._num_to_produce = min(
                self._num_to_produce,
                remaining_space // self._space_taken_per_lot if self._space_taken_per_lot > 1 else remaining_space,
            )

        # Check the source material inventory limits.
        if self._num_to_produce > 0 and len(self._bom):
            self._num_to_produce = min(
                self._num_to_produce,
                min([
                    self.facility.storage.get_product_quantity(sku_id) // consumption
                    for sku_id, consumption in self._bom.items()
                ])
            )

        # Start manufacture.
        if self._num_to_produce > 0:
            # Take source SKUs.
            source_sku_to_take = {}
            for sku_id, consumption in self._bom.items():
                source_sku_to_take[sku_id] = self._num_to_produce * consumption
            self.facility.storage.try_take_products(source_sku_to_take)

            self._products_in_pipeline[tick + self._manufacture_leading_time] += self._num_to_produce

        # Count manufacture cost.
        self._in_pipeline_quantity = sum([quantity for quantity in self._products_in_pipeline.values()])
        self._manufacture_cost = self._unit_product_cost * self._in_pipeline_quantity

    def execute_manufacture(self, tick: int) -> None:
        self._manufacture(tick)

        # Get finished products from pipeline.
        self._finished_quantity = self._products_in_pipeline.get(tick, 0)
        if self._finished_quantity > 0:
            self.facility.storage.try_add_products({self.sku_id: self._finished_quantity})
            self._products_in_pipeline.pop(tick)

    def flush_states(self) -> None:
        self.data_model.start_manufacture_quantity = self._num_to_produce
        self.data_model.in_pipeline_quantity = self._in_pipeline_quantity
        self.data_model.finished_quantity = self._finished_quantity
        self.data_model.manufacture_cost = self._manufacture_cost

    def reset(self) -> None:
        super(ManufactureUnit, self).reset()

        self._manufacture_rate = self._max_manufacture_rate
        self._products_in_pipeline.clear()

    def get_unit_info(self) -> ManufactureUnitInfo:
        return ManufactureUnitInfo(
            **super(ManufactureUnit, self).get_unit_info().__dict__,
        )


class SimpleManufactureUnit(ManufactureUnit):
    """This simple manufacture unit will ignore source sku, just generate specified number of product."""

    def __init__(
        self, id: int, data_model_name: Optional[str], data_model_index: Optional[int],
        facility: FacilityBase, parent: Union[FacilityBase, UnitBase], world: World, config: dict,
    ) -> None:
        super(SimpleManufactureUnit, self).__init__(
            id, data_model_name, data_model_index, facility, parent, world, config,
        )

    def _manufacture(self, tick: int) -> None:
        # Update num_to_produce according to limitations.
        self._num_to_produce = self._manufacture_rate * self._output_units_per_lot

        # Check the remaining space limits. TODO: confirm the remaining space setting.
        if self._num_to_produce > 0:
            remaining_space = self.facility.storage.get_product_max_remaining_space(self.sku_id)
            self._num_to_produce = min(
                self._num_to_produce,
                remaining_space // self._space_taken_per_lot if self._space_taken_per_lot > 1 else remaining_space,
            )

        # Start manufacture.
        if self._num_to_produce > 0:
            self._products_in_pipeline[tick + self._manufacture_leading_time] += self._num_to_produce

        # Count manufacture cost.
        self._in_pipeline_quantity = sum([quantity for quantity in self._products_in_pipeline.values()])
        self._manufacture_cost = self._unit_product_cost * self._in_pipeline_quantity
