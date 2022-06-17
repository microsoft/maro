# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from typing import Optional

DEFAULT_EXPIRATION_BUFFER = 3


@dataclass
class SupplyChainAction:
    id: int


@dataclass
class ConsumerAction(SupplyChainAction):
    sku_id: int
    source_id: int
    quantity: int
    vehicle_type: str
    expiration_buffer: Optional[int] = DEFAULT_EXPIRATION_BUFFER


@dataclass
class ManufactureAction(SupplyChainAction):
    manufacture_rate: int
