# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import Optional, Union

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
        facility: FacilityBase, parent: Union[FacilityBase, UnitBase], world: World, config: dict
    ) -> None:
        super(ManufactureUnit, self).__init__(id, data_model_name, data_model_index, facility, parent, world, config)

        # Source material sku and related quantity per manufacture cycle.
        self._bom: Optional[dict] = None

        # How many products in each manufacture cycle.
        self._output_units_per_lot: Optional[int] = None

        # How many units we will consume in each manufacture cycle.
        self._input_units_per_lot: int = 0

        # How many products we manufacture in current step.
        self._manufacture_quantity: int = 0

    def initialize(self) -> None:
        super(ManufactureUnit, self).initialize()

        assert isinstance(self.data_model, ManufactureDataModel)

        unit_product_cost = self.facility.skus[self.product_id].unit_product_cost
        self.data_model.initialize(unit_product_cost)

        global_sku_info = self.world.get_sku_by_id(self.product_id)

        self._bom = global_sku_info.bom
        self._output_units_per_lot = global_sku_info.output_units_per_lot

        if len(self._bom) > 0:
            self._input_units_per_lot = sum(self._bom.values())

    def _step_impl(self, tick: int) -> None:
        # TODO: Set default production rate, and produce according to it if Action not given.

        # Due to the processing in post_step(),
        # self._manufacture_quantity is set to 0 at the begining of every step.
        # Thus, there is no need to update it with None action or 0 production_rate.

        if self.action is None:
            return

        assert isinstance(self.action, ManufactureAction)

        # Try to produce production if we have positive rate.
        if self.action.production_rate > 0:
            max_number_to_procedure = min(
                self.action.production_rate * self._output_units_per_lot,
                self.facility.storage.get_product_max_remaining_space(self.product_id),
            )

            if max_number_to_procedure > 0:
                space_taken_per_cycle = self._output_units_per_lot - self._input_units_per_lot

                # Consider about the volume, we can produce all if space take per cycle <=1.
                if space_taken_per_cycle > 1:
                    max_number_to_procedure = max_number_to_procedure // space_taken_per_cycle

                # Do we have enough source material?
                for source_sku_id, source_sku_cost_number in self._bom.items():
                    source_sku_available_number = self.facility.storage.get_product_quantity(source_sku_id)

                    max_number_to_procedure = min(
                        source_sku_available_number // source_sku_cost_number,
                        max_number_to_procedure,
                    )

                    if max_number_to_procedure <= 0:
                        break

                if max_number_to_procedure > 0:
                    source_sku_to_take = {}
                    for source_sku_id, source_sku_cost_number in self._bom.items():
                        source_sku_to_take[source_sku_id] = max_number_to_procedure * source_sku_cost_number

                    self._manufacture_quantity = max_number_to_procedure
                    self.facility.storage.try_take_products(source_sku_to_take)
                    self.facility.storage.try_add_products({self.product_id: self._manufacture_quantity})

    def flush_states(self) -> None:
        if self._manufacture_quantity > 0:
            self.data_model.manufacture_quantity = self._manufacture_quantity

    def post_step(self, tick: int) -> None:
        if self._manufacture_quantity > 0:
            self.data_model.manufacture_quantity = 0
            self._manufacture_quantity = 0

        # NOTE: call super at last, since it will clear the action.
        super(ManufactureUnit, self).post_step(tick)

    def reset(self) -> None:
        super(ManufactureUnit, self).reset()

        # Reset status in Python side.
        self._manufacture_quantity = 0

    def get_unit_info(self) -> ManufactureUnitInfo:
        return ManufactureUnitInfo(
            **super(ManufactureUnit, self).get_unit_info().__dict__,
        )


class SimpleManufactureUnit(ManufactureUnit):
    """This simple manufacture unit will ignore source sku, just generate specified number of product."""

    def __init__(
        self, id: int, data_model_name: Optional[str], data_model_index: Optional[int],
        facility: FacilityBase, parent: Union[FacilityBase, UnitBase], world: World, config: dict
    ) -> None:
        super(SimpleManufactureUnit, self).__init__(
            id, data_model_name, data_model_index, facility, parent, world, config
        )

    def _step_impl(self, tick: int) -> None:
        if self.action is None:
            return

        assert isinstance(self.action, ManufactureAction)

        # Try to produce production if we have positive rate.
        if self.action.production_rate > 0:
            production_rate = self.action.production_rate

            self._manufacture_quantity = min(
                self.facility.storage.get_product_max_remaining_space(self.product_id),
                production_rate,
            )

            if self._manufacture_quantity > 0:
                self.facility.storage.try_add_products({self.product_id: self._manufacture_quantity})
