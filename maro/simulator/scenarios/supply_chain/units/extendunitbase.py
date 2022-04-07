# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass

from .unitbase import BaseUnitInfo, UnitBase


@dataclass
class ExtendUnitInfo(BaseUnitInfo):
    product_id: int


class ExtendUnitBase(UnitBase):
    """A base of sku related unit."""

    def __init__(self) -> None:
        super(ExtendUnitBase, self).__init__()

        # Product id (sku id), 0 means invalid.
        self.product_id: int = 0

    def initialize(self):
        super(ExtendUnitBase, self).initialize()

        if self.data_model is not None:
            self.data_model.set_product_id(self.product_id, self.parent.id)

    def get_unit_info(self) -> ExtendUnitInfo:
        return ExtendUnitInfo(
            **super(ExtendUnitBase, self).get_unit_info().__dict__,
            product_id=self.product_id,
        )
