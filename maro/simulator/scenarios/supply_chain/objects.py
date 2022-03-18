# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from dataclasses import dataclass


@dataclass
class SkuMeta:
    id: int
    name: str
    output_units_per_lot: int
    bom: dict = None


@dataclass
class SkuInfo:
    id: int
    init_stock: int
    backlog_ratio: float = 0.0
    cost: int = 10
    price: int = 10
    product_unit_cost: int = 1
    production_rate: float = 1.0
    sale_gamma: int = 100
    type: str = None
    vlt: int = 1  # TODO: update the vlt related code
    service_level: float = 0.95
