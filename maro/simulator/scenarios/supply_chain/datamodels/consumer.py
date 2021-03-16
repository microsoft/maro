# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.backends.backend import AttributeType
from maro.backends.frame import node, NodeAttribute

from .base import DataModelBase


# NOTE: one sku one consumer
@node("consumer")
class ConsumerDataModel(DataModelBase):
    # reward states
    # from config
    order_cost = NodeAttribute(AttributeType.UInt)
    total_purchased = NodeAttribute(AttributeType.UInt)
    total_received = NodeAttribute(AttributeType.UInt)

    # action states
    product_id = NodeAttribute(AttributeType.UInt)
    source_id = NodeAttribute(AttributeType.UInt)
    quantity = NodeAttribute(AttributeType.UInt)
    vlt = NodeAttribute(AttributeType.UShort)

    # id of upstream facilities.
    sources = NodeAttribute(AttributeType.UInt, 1, is_list=True)

    # per tick states

    # snapshots["consumer"][hist_len::"purchased"] equals to original latest_consumptions
    purchased = NodeAttribute(AttributeType.UInt)
    received = NodeAttribute(AttributeType.UInt)
    order_product_cost = NodeAttribute(AttributeType.UInt)

    def __init__(self):
        super(ConsumerDataModel, self).__init__()

        self._order_cost = 0
        self._product_id = 0

    def initialize(self, order_cost=0, product_id=0):
        self._order_cost = order_cost
        self._product_id = product_id

        self.reset()

    def reset(self):
        super(ConsumerDataModel, self).reset()

        self.order_cost = self._order_cost
        self.product_id = self._product_id
