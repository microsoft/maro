from .base import DataModelBase

from maro.backends.frame import node, NodeBase, NodeAttribute
from maro.backends.backend import AttributeType


# NOTE: one sku one seller
@node("seller")
class SellerDataModel(DataModelBase):
    # reward states
    total_sold = NodeAttribute(AttributeType.Int)

    # action states
    unit_price = NodeAttribute(AttributeType.Int)

    #
    sale_gamma = NodeAttribute(AttributeType.Int)

    # what we will sell
    product_id = NodeAttribute(AttributeType.Int)

    # per tick state, we can use this to support "sale hist" feature in original code.
    # original there is only sold state, we add a demand here
    demand = NodeAttribute(AttributeType.Int)
    sold = NodeAttribute(AttributeType.Int)

    def __init__(self):
        super(SellerDataModel, self).__init__()

        self._unit_price = 0
        self._sale_gamma = 0
        self._product_id = 0

    def initialize(self, configs: dict):
        if configs is not None:
            self._unit_price = configs.get("unit_price", 0)
            self._sale_gamma = configs.get("sale_gamma", 0)
            self._product_id = configs.get("product_id", 0)

            self.reset()

    def reset(self):
        super(SellerDataModel, self).reset()

        self.unit_price = self._unit_price
        self.product_id = self._product_id
        self.sale_gamma = self._sale_gamma
