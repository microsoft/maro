# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import typing
from dataclasses import dataclass

if typing.TYPE_CHECKING:
    from maro.simulator.scenarios.supply_chain.facilities import FacilityBase


@dataclass
class Order:
    destination: FacilityBase
    product_id: int
    quantity: int
    vehicle_type: str
