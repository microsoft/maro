# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.backends.backend import AttributeType
from maro.backends.frame import node, NodeAttribute

from .base import DataModelBase


@node("facility")
class FacilityDataModel(DataModelBase):
    test = NodeAttribute(AttributeType.UInt)

    def __init__(self):
        super(FacilityDataModel, self).__init__()

    def initialize(self, configs: dict):
        self.reset()

    def reset(self):
        super(FacilityDataModel, self).reset()
