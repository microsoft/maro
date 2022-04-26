# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.backends.backend import AttributeType
from maro.backends.frame import NodeAttribute, node

from .extend import ExtendDataModel


@node("manufacture")
class ManufactureDataModel(ExtendDataModel):
    """Data model for manufacture unit."""
    start_manufacture_quantity = NodeAttribute(AttributeType.UInt)
    in_pipeline_quantity = NodeAttribute(AttributeType.UInt)
    finished_quantity = NodeAttribute(AttributeType.UInt)

    manufacture_cost = NodeAttribute(AttributeType.Float)

    def __init__(self) -> None:
        super(ManufactureDataModel, self).__init__()

    def initialize(self) -> None:
        self.reset()

    def reset(self) -> None:
        super(ManufactureDataModel, self).reset()
