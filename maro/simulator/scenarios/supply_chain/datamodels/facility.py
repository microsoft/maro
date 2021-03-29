# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.backends.backend import AttributeType
from maro.backends.frame import NodeAttribute, node

from .base import DataModelBase


@node("facility")
class FacilityDataModel(DataModelBase):
    """Data model for facilities.

    NOTE:
        Not in use for now.
    """
    balance_sheet = NodeAttribute(AttributeType.UInt)

    def __init__(self):
        super(FacilityDataModel, self).__init__()
