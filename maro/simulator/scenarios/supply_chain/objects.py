# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import typing
from dataclasses import dataclass, field
from typing import Dict, Optional

if typing.TYPE_CHECKING:
    from .facilities import FacilityBase


DEFAULT_SUB_STORAGE_ID = 0


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
    price: float  # Would be used both in SellerUnit (to end customers) and DistributionUnit (to downstream facilities)

    # Storage config
    init_stock: int
    unit_storage_cost: Optional[float] = None
    sub_storage_id: int = DEFAULT_SUB_STORAGE_ID  # TODO: decide whether it could be a default setting
    storage_upper_bound: Optional[int] = None  # TODO: Or split the storage directly?

    # Manufacture config
    has_manufacture: bool = False  # To indicate whether the ProductUnit has a ManufactureUnit or not
    unit_product_cost: Optional[float] = None
    manufacture_rate: Optional[int] = None  # The initial production rate.
    # manufacture_leading_time: Optional[int] = None

    # Consumer config
    has_consumer: bool = False  # To indicate whether the ProductUnit has a ConsumerUnit or not
    # SKU specific one would be used if set, else the one for its facility would be used.
    unit_order_cost: Optional[float] = None
    # Seller config
    has_seller: bool = False  # To indicate whether the SellerUnit has a ConsumerUnit or not
    sale_gamma: Optional[int] = None
    backlog_ratio: Optional[float] = None

    # Distribution config
    # SKU specific one would be used if set, else the one for its facility would be used.
    unit_delay_order_penalty: Optional[float] = None
    # For policy only
    service_level: float = 0.95


@dataclass
class VendorLeadingTimeInfo:
    src_facility: FacilityBase  # TODO: change to facility id?
    vehicle_type: str
    vlt: int
    unit_transportation_cost: float


@dataclass
class LeadingTimeInfo:
    dest_facility: FacilityBase  # TODO: change to facility id?
    vehicle_type: str
    vlt: int
    unit_transportation_cost: float


@dataclass
class SupplyChainEntity:
    id: int
    class_type: type
    skus: Optional[SkuInfo]
    facility_id: int
    parent_id: Optional[int]


@dataclass
class SubStorageConfig:
    id: int
    capacity: int = 100  # TODO: Is it a MUST config or could it be default?
    unit_storage_cost: int = 1


def parse_storage_config(config: dict) -> Dict[int, SubStorageConfig]:  # TODO: here or in parser
    if not isinstance(config, list):
        id = config.get("id", DEFAULT_SUB_STORAGE_ID)
        return {id: SubStorageConfig(id=id, **config)}
    return {SubStorageConfig(**cfg).id: SubStorageConfig(**cfg) for cfg in config}
