from .base import DataModelBase

from maro.backends.frame import node, NodeBase, NodeAttribute
from maro.backends.backend import AttributeType


# NOTE: one sku one consumer
@node("consumer")
class ConsumerDataModel(DataModelBase):
    # reward states
    # from config
    order_cost = NodeAttribute(AttributeType.Int)
    total_purchased = NodeAttribute(AttributeType.Int)
    total_received = NodeAttribute(AttributeType.Int)

    # action states
    consumer_product_id = NodeAttribute(AttributeType.Int)
    consumer_source_id = NodeAttribute(AttributeType.Int)
    consumer_quantity = NodeAttribute(AttributeType.Int)
    consumer_vlt = NodeAttribute(AttributeType.Int)

    # id of upstream facilities.
    sources = NodeAttribute(AttributeType.Int, 1, is_list=True)

    # per tick states

    # snapshots["consumer"][hist_len::"purchased"] equals to original latest_consumptions
    purchased = NodeAttribute(AttributeType.Int)
    received = NodeAttribute(AttributeType.Int)
    order_product_cost = NodeAttribute(AttributeType.Int)

    def __init__(self):
        super(ConsumerDataModel, self).__init__()

        self._order_cost = 0
        self._consumer_product_id = 0

    def initialize(self, configs: dict):
        if configs is not None:
            self._order_cost = configs["order_cost"]
            self._consumer_product_id = configs["consumer_product_id"]

            self.reset()

    def reset(self):
        super(ConsumerDataModel, self).reset()

        self.order_cost = self._order_cost
        self.consumer_product_id = self._consumer_product_id
