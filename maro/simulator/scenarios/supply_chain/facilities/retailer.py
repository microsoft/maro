# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from collections import namedtuple
from typing import List

from maro.simulator.scenarios.supply_chain.units import ProductUnit, StorageUnit

from .facility import FacilityBase


class RetailerFacility(FacilityBase):
    """Retail facility used to generate order from upstream, and sell products by demand."""

    # Product unit list of this facility.
    products: List[ProductUnit]

    # Storage unit of this facility.
    storage: StorageUnit

    def __init__(self):
        super().__init__()
