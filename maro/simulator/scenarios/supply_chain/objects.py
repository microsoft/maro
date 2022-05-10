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

    """
    Manufacture configs:
    - has_manufacture: To indicate whether the ProductUnit has a ManufactureUnit or not.
    - unit_product_cost: Must set if has_manufacture.
        . Per unit, per tick.
        . The manufacture/production cost would be: unit_product_cost * manufacture_rate * manufacture_leading_time.
    - max_manufacture_rate: Must set if has_manufacture.
        . The manufacture capacity, or said the throughput of the manufacture pipeline.
        . It is the upper bound of valid manufacture_rate could be set in the manufacture action.
        . Note that it is the maximal product quantity we get at the end of a manufacture cycle (after a whole leading
        time duration, not for each tick.)
    - manufacture_leading_time: Must set if has_manufacture.
        . How many ticks would it take to produce products.
    """
    has_manufacture: bool = False
    unit_product_cost: Optional[float] = None
    max_manufacture_rate: Optional[int] = None
    manufacture_leading_time: int = 0  # TODO: update default value after BE updated

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
    unit_transportation_cost: float  # Unit cost by day, the whole trip cost would be: unit cost * (vlt + 1)

    def __repr__(self) -> str:
        return (
            f"Vehicle {self.vehicle_type} from facility {self.src_facility.name}: "
            f"({self.unit_transportation_cost}, {self.vlt} days)"
        )


@dataclass
class LeadingTimeInfo:
    dest_facility: FacilityBase  # TODO: change to facility id?
    vehicle_type: str
    vlt: int
    unit_transportation_cost: float  # Unit cost by day, the whole trip cost would be: unit cost * (vlt + 1)

    def __repr__(self) -> str:
        return (
            f"Vehicle {self.vehicle_type} to facility {self.dest_facility.name}: "
            f"({self.unit_transportation_cost}, {self.vlt} days)"
        )


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


def parse_storage_config(config: dict) -> Dict[int, SubStorageConfig]:
    if not isinstance(config, list):
        id = config.get("id", DEFAULT_SUB_STORAGE_ID)
        return {id: SubStorageConfig(id=id, **config)}
    return {SubStorageConfig(**cfg).id: SubStorageConfig(**cfg) for cfg in config}
