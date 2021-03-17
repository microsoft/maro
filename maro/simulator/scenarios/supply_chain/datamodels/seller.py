# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.backends.backend import AttributeType
from maro.backends.frame import node, NodeAttribute

from .skumodel import SkuDataModel


# NOTE: one sku one seller
@node("seller")
class SellerDataModel(SkuDataModel):
    # reward states
    total_sold = NodeAttribute(AttributeType.UInt)

    # from config
    backlog_ratio = NodeAttribute(AttributeType.Float)

    # action states
    unit_price = NodeAttribute(AttributeType.UInt)

    #
    sale_gamma = NodeAttribute(AttributeType.UInt)

    # per tick state, we can use this to support "sale hist" feature in original code.
    # original there is only sold state, we add a demand here
    demand = NodeAttribute(AttributeType.UInt)
    sold = NodeAttribute(AttributeType.UInt)

    def __init__(self):
        super(SellerDataModel, self).__init__()

        self._unit_price = 0
        self._sale_gamma = 0
        self._product_id = 0
        self._backlog_ratio = 0

    def initialize(self, unit_price: int = 0, sale_gamma: int = 0, product_id: int = 0, backlog_ratio: int = 0):
        self._unit_price = unit_price
        self._sale_gamma = sale_gamma
        self._product_id = product_id
        self._backlog_ratio = backlog_ratio

        self.reset()

    def reset(self):
        super(SellerDataModel, self).reset()

        self.unit_price = self._unit_price
        self.product_id = self._product_id
        self.sale_gamma = self._sale_gamma
        self.backlog_ratio = self._backlog_ratio
