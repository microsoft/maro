# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.backends.backend import AttributeType
from maro.backends.frame import NodeAttribute, node

from .skumodel import SkuDataModel


@node("manufacture")
class ManufactureDataModel(SkuDataModel):
    """Data model for manufacture unit."""
    # Storage related to this manufacture unit, for easy state retrieving.
    storage_id = NodeAttribute(AttributeType.UInt)

    # Cost to produce one output production.
    product_unit_cost = NodeAttribute(AttributeType.UInt)

    # Number per tick, different with original manufacturing cost, we just provide number, and cost
    # user can determine how to calculate the cost.
    manufacturing_number = NodeAttribute(AttributeType.UInt)

    production_rate = NodeAttribute(AttributeType.UInt)

    def __init__(self):
        super(ManufactureDataModel, self).__init__()
        self._product_unit_cost = 0
        self._storage_id = 0

    def initialize(self, product_unit_cost: int = 1, storage_id: int = 0):
        """Initialize data model, used to assign value after frame reset.

        Args:
            product_unit_cost (int): Cost per unit, from configuration files.
            storage_id (int): Storage id of this manufacture's facility.
        """
        self._product_unit_cost = product_unit_cost
        self._storage_id = storage_id

        self.reset()

    def reset(self):
        super(ManufactureDataModel, self).reset()

        self.product_unit_cost = self._product_unit_cost
        self.storage_id = self._storage_id
