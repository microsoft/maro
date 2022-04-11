# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import typing
from dataclasses import dataclass, field
from typing import Dict, Optional

from .units.storage import DEFAULT_SUB_STORAGE_ID

if typing.TYPE_CHECKING:
    from .facilities import FacilityBase


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
    sub_storage_id: int = DEFAULT_SUB_STORAGE_ID  # TODO: decide whether it could be a default setting
    storage_upper_bound: Optional[int] = None  # TODO: Or separate the storage directly?
    # Manufacture config
    has_manufacture: bool = False  # To indicate whether the ProductUnit has a ManufactureUnit or not
    unit_product_cost: Optional[float] = None
    production_rate: Optional[int] = None  # The initial production rate.
    # Consumer config
    has_consumer: bool = False  # To indicate whether the ProductUnit has a ConsumerUnit or not
    unit_order_cost: Optional[float] = None  # SKU specific one would be used if set, else the one for its facility would be used.
    # Seller config
    has_seller: bool = False  # To indicate whether the SellerUnit has a ConsumerUnit or not
    sale_gamma: Optional[int] = None
    backlog_ratio: Optional[float] = None
    # Distribution config
    unit_delay_order_penalty: Optional[float] = None  # SKU specific one would be used if set, else the one for its facility would be used.
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

    @property
    def is_facility(self) -> bool:
        from .facilities import FacilityBase
        return issubclass(self.class_type, FacilityBase)
