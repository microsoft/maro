# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.backends.frame import node

from .facility import FacilityDataModel


@node("retailer")
class RetailerFacilityDataModel(FacilityDataModel):

    def __init__(self):
        super(RetailerFacilityDataModel, self).__init__()

    def reset(self):
        super(RetailerFacilityDataModel, self).reset()
