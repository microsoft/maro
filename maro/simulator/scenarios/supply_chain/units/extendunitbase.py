# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from .unitbase import UnitBase


class ExtendUnitBase(UnitBase):
    """A base of sku related unit."""

    # Product id (sku id), 0 means invalid.
    product_id: int = 0

    def initialize(self):
        super(ExtendUnitBase, self).initialize()

        if self.data_model is not None:
            self.data_model.set_product_id(self.product_id, self.parent.id)

    def get_unit_info(self) -> dict:
        info = super(ExtendUnitBase, self).get_unit_info()

        info["sku_id"] = self.product_id

        return info
