# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from typing import NamedTuple


class ConsumerAction(NamedTuple):
    id: int
    product_id: int
    source_id: int
    quantity: int
    vlt: int
    reward_discount: float


class ManufactureAction(NamedTuple):
    id: int
    production_rate: float
