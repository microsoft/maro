# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.backends.backend import AttributeType
from maro.backends.frame import NodeAttribute, node

from .base import DataModelBase


@node("vehicle")
class VehicleDataModel(DataModelBase):
    # Number of product.
    payload = NodeAttribute(AttributeType.UInt)

    unit_transport_cost = NodeAttribute(AttributeType.Float)

    def __init__(self) -> None:
        super(VehicleDataModel, self).__init__()

        self._unit_transport_cost = 1

    def initialize(self, unit_transport_cost: int = 1) -> None:
        self._unit_transport_cost = unit_transport_cost

        self.reset()

    def reset(self) -> None:
        super(VehicleDataModel, self).reset()

        self.unit_transport_cost = self._unit_transport_cost
