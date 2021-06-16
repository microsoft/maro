# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.backends.frame import node

from .facility import FacilityDataModel


@node("warehouse")
class WarehouseFacilityDataModel(FacilityDataModel):

    def __init__(self):
        super(WarehouseFacilityDataModel, self).__init__()

    def reset(self):
        super(WarehouseFacilityDataModel, self).reset()
