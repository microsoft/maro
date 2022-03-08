# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.backends.backend import AttributeType
from maro.backends.frame import NodeAttribute, node

from .extend import ExtendDataModel


@node("product")
class ProductDataModel(ExtendDataModel):
    price = NodeAttribute(AttributeType.Float)
    distribution_check_order = NodeAttribute(AttributeType.UInt)
    distribution_transport_cost = NodeAttribute(AttributeType.Float)
    distribution_delay_order_penalty = NodeAttribute(AttributeType.Float)

    def __init__(self) -> None:
        super(ProductDataModel, self).__init__()

        self._price = 0

    def initialize(self, price: float) -> None:
        self._price = price

        self.reset()

    def reset(self) -> None:
        super(ProductDataModel, self).reset()

        self.price = self._price
