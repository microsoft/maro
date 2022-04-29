# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.backends.backend import AttributeType
from maro.backends.frame import NodeAttribute, node

from .extend import ExtendDataModel


@node("manufacture")
class ManufactureDataModel(ExtendDataModel):
    """Data model for manufacture unit."""
    # Can be updated in ManufactureUnit._manufacture(), called in post_step()
    manufacture_cost = NodeAttribute(AttributeType.Float)
    start_manufacture_quantity = NodeAttribute(AttributeType.UInt)
    in_pipeline_quantity = NodeAttribute(AttributeType.UInt)

    # Can be updated in ManufactureUnit.post_step()
    finished_quantity = NodeAttribute(AttributeType.UInt)

    def __init__(self) -> None:
        super(ManufactureDataModel, self).__init__()

    def initialize(self) -> None:
        self.reset()

    def reset(self) -> None:
        super(ManufactureDataModel, self).reset()
