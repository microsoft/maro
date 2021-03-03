from .base import DataModelBase

from maro.backends.frame import node, NodeBase, NodeAttribute
from maro.backends.backend import AttributeType


@node("storage")
class StorageDataModel(DataModelBase):
    unit_storage_cost = NodeAttribute(AttributeType.Int)
    remaining_space = NodeAttribute(AttributeType.Int)
    capacity = NodeAttribute(AttributeType.Int)

    # original stock_levels, used to save product and its number
    product_list = NodeAttribute(AttributeType.Int, 1, is_list=True)
    product_number = NodeAttribute(AttributeType.Int, 1, is_list=True)

    def __init__(self):
        self._unit_storage_cost = 0
        self._capacity = 0

    def initialize(self, configs):
        if configs is not None:
            self._unit_storage_cost = configs.get("unit_storage_cost", 0)
            self._capacity = configs.get("capacity", 0)

            self.reset()

    def reset(self):
        self.unit_storage_cost = self._unit_storage_cost
        self.capacity = self._capacity

        self.remaining_space = self._capacity

        self.product_number[:] = 0
