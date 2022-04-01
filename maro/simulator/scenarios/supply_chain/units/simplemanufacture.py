# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import typing
from typing import Optional, Union

from maro.simulator.scenarios.supply_chain.actions import ManufactureAction

from .manufacture import ManufactureUnit
from .unitbase import UnitBase

if typing.TYPE_CHECKING:
    from maro.simulator.scenarios.supply_chain.facilities import FacilityBase
    from maro.simulator.scenarios.supply_chain.world import World


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
