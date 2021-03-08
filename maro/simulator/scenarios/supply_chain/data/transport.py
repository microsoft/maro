from .base import DataModelBase
from maro.backends.frame import node, NodeBase, NodeAttribute
from maro.backends.backend import AttributeType


@node("transport")
class TransportDataModel(DataModelBase):
    # Id of current entity
    source = NodeAttribute(AttributeType.Int)

    # Id of target entity.
    destination = NodeAttribute(AttributeType.Int)

    # Number of product.
    payload = NodeAttribute(AttributeType.Int)

    # Index of product.
    product_id = NodeAttribute(AttributeType.Int)

    requested_quantity = NodeAttribute(AttributeType.Int)

    # Patient to wait for products ready.
    patient = NodeAttribute(AttributeType.Int)

    # Steps to destination.
    steps = NodeAttribute(AttributeType.Int)

    # Current location on the way, equal to step means arrive at destination.
    location = NodeAttribute(AttributeType.Int)

    # for debug
    position = NodeAttribute(AttributeType.Int, 2)

    def __init__(self):
        super(TransportDataModel, self).__init__()

        self._patient = 0

    def initialize(self, configs: dict):
        if configs is not None:
            self._patient = configs.get("patient", 100)

            self.reset()

    def reset(self):
        super(TransportDataModel, self).reset()

        self.patient = self._patient
        self.position[:] = -1
