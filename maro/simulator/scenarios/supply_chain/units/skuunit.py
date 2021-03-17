# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from .unitbase import UnitBase


class SkuUnit(UnitBase):
    """A sku related unit."""

    # Product id (sku id), 0 means invalid.
    product_id: int = 0

    def initialize(self):
        super(SkuUnit, self).initialize()

        if self.data_model is not None:
            self.data_model.set_product_id(self.product_id, self.parent.id)
