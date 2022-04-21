# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.backends.backend import AttributeType
from maro.backends.frame import NodeAttribute, node

from .extend import ExtendDataModel


@node("consumer")
class ConsumerDataModel(ExtendDataModel):
    """Data model for consumer unit."""
    purchased = NodeAttribute(AttributeType.UInt)
    received = NodeAttribute(AttributeType.UInt)

    order_product_cost = NodeAttribute(AttributeType.UInt)  # order.quantity * upstream.price
    order_base_cost = NodeAttribute(AttributeType.Float)  # order.quantity * unit_order_cost

    latest_consumptions = NodeAttribute(AttributeType.Float)


    def __init__(self) -> None:
        super(ConsumerDataModel, self).__init__()

    def initialize(self) -> None:
        self.reset()

    def reset(self) -> None:
        super(ConsumerDataModel, self).reset()
