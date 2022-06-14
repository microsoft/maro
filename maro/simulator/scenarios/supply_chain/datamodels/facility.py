# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.backends.frame import node

from .base import DataModelBase


@node("facility")
class FacilityDataModel(DataModelBase):
    def __init__(self) -> None:
        super(FacilityDataModel, self).__init__()

    def reset(self) -> None:
        super(FacilityDataModel, self).reset()
