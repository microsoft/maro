# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.backends.backend import AttributeType
from maro.backends.frame import NodeAttribute, node

from .extend import ExtendDataModel


@node("consumer")
class ConsumerDataModel(ExtendDataModel):
    """Data model for consumer unit."""

    # Can be updated in on_order_reception() <- called by DistributionUnit.post_step().
    received = NodeAttribute(AttributeType.UInt)

    # Below 4 attributes, can be updated in ConsumerUnit.on_action_received() <- triggered by ConsumerAction.
    purchased = NodeAttribute(AttributeType.UInt)
    order_product_cost = NodeAttribute(AttributeType.Float)  # order.quantity * upstream.price
    order_base_cost = NodeAttribute(AttributeType.Float)  # order.quantity * unit_order_cost

    latest_consumptions = NodeAttribute(AttributeType.Float)

    in_transit_quantity = NodeAttribute(AttributeType.UInt)

    def __init__(self) -> None:
        super(ConsumerDataModel, self).__init__()

    def initialize(self) -> None:
        self.reset()

    def reset(self) -> None:
        super(ConsumerDataModel, self).reset()
