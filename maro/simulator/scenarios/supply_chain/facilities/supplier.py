# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from collections import namedtuple
from typing import List

from maro.simulator.scenarios.supply_chain.units import DistributionUnit, ProductUnit, StorageUnit

from .facility import FacilityBase


class SupplierFacility(FacilityBase):
    """Supplier facilities used to produce products with material products."""

    # Storage unit of this facility.
    storage: StorageUnit

    # Distribution unit of this facility.
    distribution: DistributionUnit

    # Product unit list of this facility.
    products: List[ProductUnit]

    def __init__(self):
        super(SupplierFacility, self).__init__()
