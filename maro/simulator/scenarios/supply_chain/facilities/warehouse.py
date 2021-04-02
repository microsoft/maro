# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from collections import namedtuple
from typing import List

from maro.simulator.scenarios.supply_chain.units import DistributionUnit, ProductUnit, StorageUnit

from .facility import FacilityBase


class WarehouseFacility(FacilityBase):
    """Warehouse facility that used to storage products, composed with storage, distribution and product units."""

    # Storage unit for this facility, must be a sub class of StorageUnit.
    storage: StorageUnit = None

    # Distribution unit for this facility.
    distribution: DistributionUnit = None

    # Product unit list for this facility.
    products: List[ProductUnit] = None

    def __init__(self):
        super().__init__()
