# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.backends.backend import AttributeType
from maro.backends.frame import NodeAttribute, node

from .base import DataModelBase


@node("vehicle")
class VehicleDataModel(DataModelBase):
    # Number of product.
    payload = NodeAttribute(AttributeType.UInt)

    # Patient to wait for products ready.
    patient = NodeAttribute(AttributeType.UInt)

    def __init__(self):
        super(VehicleDataModel, self).__init__()

        self._patient = 0

    def initialize(self, patient: int = 100):
        self._patient = patient

        self.reset()

    def reset(self):
        super(VehicleDataModel, self).reset()

        self.patient = self._patient
