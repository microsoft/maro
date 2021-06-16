# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.backends.frame import node

from .facility import FacilityDataModel


@node("supplier")
class SupplierFacilityDataModel(FacilityDataModel):

    def __init__(self):
        super(SupplierFacilityDataModel, self).__init__()

    def reset(self):
        super(SupplierFacilityDataModel, self).reset()
