# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from .. import ManufactureAction
from .manufacture import ManufactureUnit


class SimpleManufactureUnit(ManufactureUnit):
    """This simple manufacture unit will ignore source sku, just generate specified number of product."""

    def _step_impl(self, tick: int) -> None:
        # Try to produce production if we have positive rate.
        self.manufacture_number = 0

        assert self.action is None or isinstance(self.action, ManufactureAction)

        if self.action is not None and self.action.production_rate > 0:
            production_rate = self.action.production_rate

            sku_num = len(self.facility.skus)
            unit_num_upper_bound = self.facility.storage.capacity // sku_num
            current_product_number = self.facility.storage.get_product_number(self.product_id)
            self.manufacture_number = max(0, min(unit_num_upper_bound - current_product_number, production_rate))

            if self.manufacture_number > 0:
                self.facility.storage.try_add_products({self.product_id: self.manufacture_number})
