# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass, field
from typing import Dict, Optional

from .units.storage import DEFAULT_SUB_STORAGE_ID


@dataclass
class SkuMeta:
    id: int
    name: str
    output_units_per_lot: int = 1
    bom: Dict[int, int] = field(default_factory=dict)  # Key: SKU id, Value: required quantity per lot


@dataclass
class SkuInfo:
    id: int
    name: str
    init_stock: int
    sub_storage_id: int = DEFAULT_SUB_STORAGE_ID  # TODO: decide whether it could be a default setting
    storage_upper_bound: Optional[int] = None  # TODO: Or separate the storage directly?
    backlog_ratio: float = 0.0
    cost: int = 10
    price: int = 10
    product_unit_cost: int = 1
    production_rate: float = 1.0
    sale_gamma: int = 100
    type: str = None
    vlt: int = 1  # TODO: update the vlt related code
    service_level: float = 0.95
