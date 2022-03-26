# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.simulator.scenarios.supply_chain.actions import ManufactureAction

from .manufacture import ManufactureUnit


class SimpleManufactureUnit(ManufactureUnit):
    """This simple manufacture unit will ignore source sku, just generate specified number of product."""

    def __init__(self) -> None:
        super(SimpleManufactureUnit, self).__init__()

    def _step_impl(self, tick: int) -> None:
        if self.action is None:
            return

        assert isinstance(self.action, ManufactureAction)

        # Try to produce production if we have positive rate.
        if self.action.production_rate > 0:
            production_rate = self.action.production_rate

            self._manufacture_quantity = min(
                self.facility.storage.get_product_max_remaining_space(self.product_id),
                production_rate
            )

            if self._manufacture_quantity > 0:
                self.facility.storage.try_add_products({self.product_id: self._manufacture_quantity})
