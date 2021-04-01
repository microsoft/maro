# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.backends.backend import AttributeType
from maro.backends.frame import NodeAttribute, node

from .skumodel import SkuDataModel


@node("manufacture")
class ManufactureDataModel(SkuDataModel):
    """Data model for manufacture unit."""
    # Number per tick, different with original manufacturing cost, we just provide number, and cost
    # user can determine how to calculate the cost.
    manufacturing_number = NodeAttribute(AttributeType.UInt)

    def __init__(self):
        super(ManufactureDataModel, self).__init__()

    def reset(self):
        super(ManufactureDataModel, self).reset()
