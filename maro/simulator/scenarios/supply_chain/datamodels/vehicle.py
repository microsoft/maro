# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.backends.backend import AttributeType
from maro.backends.frame import NodeAttribute, node

from .base import DataModelBase


@node("vehicle")
class VehicleDataModel(DataModelBase):
    # Id of current entity
    source = NodeAttribute(AttributeType.UInt)

    # Id of target entity.
    destination = NodeAttribute(AttributeType.UInt)

    # Number of product.
    payload = NodeAttribute(AttributeType.UInt)

    # Index of product.
    product_id = NodeAttribute(AttributeType.UInt)

    requested_quantity = NodeAttribute(AttributeType.UInt)

    # Patient to wait for products ready.
    patient = NodeAttribute(AttributeType.UInt)

    # Steps to destination.
    steps = NodeAttribute(AttributeType.UInt)

    # from config
    unit_transport_cost = NodeAttribute(AttributeType.UInt)

    position = NodeAttribute(AttributeType.Int, 2)

    def __init__(self):
        super(VehicleDataModel, self).__init__()

        self._patient = 0
        self._unit_transport_cost = 0

    def initialize(self, patient: int = 100, unit_transport_cost: int = 1):
        self._patient = patient
        self._unit_transport_cost = unit_transport_cost

        self.reset()

    def reset(self):
        super(VehicleDataModel, self).reset()

        self.unit_transport_cost = self._unit_transport_cost
        self.patient = self._patient
