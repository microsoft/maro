from .base import DataModelBase

from maro.backends.frame import node, NodeBase, NodeAttribute
from maro.backends.backend import AttributeType


@node("manufacture")
class ManufactureDataModel(DataModelBase):
    # storage related to this manufacture unit, for easy state retrieving.
    storage_id = NodeAttribute(AttributeType.Int)

    # cost to produce one output production
    product_unit_cost = NodeAttribute(AttributeType.Int)

    # cost per tick, different with original manufacturing cost, we just provide number, and cost
    # user can determine how to calculate the cost
    manufacturing_number = NodeAttribute(AttributeType.Int)

    # what we will produce
    output_product_id = NodeAttribute(AttributeType.Int)

    # original from config, then updated by action
    production_rate = NodeAttribute(AttributeType.Int)

    def __init__(self):
        super(ManufactureDataModel, self).__init__()
        self._output_product_id = 0
        self._production_rate = 0
        self._storage_id = 0

    def initialize(self, configs: dict):
        if configs is not None:
            self._output_product_id = configs["output_product_id"]
            self._production_rate = configs.get("production_rate", 1)
            self._storage_id = configs["storage_id"]

            self.reset()

    def reset(self):
        super(ManufactureDataModel, self).reset()

        self.output_product_id = self._output_product_id
        self.production_rate = self._production_rate
        self.storage_id = self._storage_id
