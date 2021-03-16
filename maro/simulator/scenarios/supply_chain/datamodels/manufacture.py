# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.backends.backend import AttributeType
from maro.backends.frame import node, NodeAttribute

from .base import DataModelBase


@node("manufacture")
class ManufactureDataModel(DataModelBase):
    # storage related to this manufacture unit, for easy state retrieving.
    storage_id = NodeAttribute(AttributeType.UInt)

    # cost to produce one output production
    product_unit_cost = NodeAttribute(AttributeType.UInt)

    # number per tick, different with original manufacturing cost, we just provide number, and cost
    # user can determine how to calculate the cost
    manufacturing_number = NodeAttribute(AttributeType.UInt)

    # what we will produce
    product_id = NodeAttribute(AttributeType.UInt)

    # original from config, then updated by action
    production_rate = NodeAttribute(AttributeType.UInt)

    def __init__(self):
        super(ManufactureDataModel, self).__init__()
        self._output_product_id = 0
        self._product_unit_cost = 0
        self._storage_id = 0

    def initialize(self, output_product_id: int = 0, product_unit_cost: int = 1, storage_id: int = 0):
        self._output_product_id = output_product_id
        self._product_unit_cost = product_unit_cost
        self._storage_id = storage_id

        self.reset()

    def reset(self):
        super(ManufactureDataModel, self).reset()

        self.product_id = self._output_product_id
        self.product_unit_cost = self._product_unit_cost
        self.storage_id = self._storage_id
