# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from .. import ManufactureAction
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

            sku_num = len(self.facility.skus)
            unit_num_upper_bound = self.facility.storage.capacity // sku_num
            current_product_quantity = self.facility.storage.get_product_quantity(self.product_id)
            self._manufacture_number = max(0, min(unit_num_upper_bound - current_product_quantity, production_rate))

            if self._manufacture_number > 0:
                self.facility.storage.try_add_products({self.product_id: self._manufacture_number})
