# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from __future__ import annotations

import typing
from typing import NamedTuple

if typing.TYPE_CHECKING:
    from maro.simulator.scenarios.supply_chain import FacilityBase


class Order(NamedTuple):
    destination: FacilityBase
    product_id: int
    quantity: int
    vlt: int
