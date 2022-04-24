# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import typing
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from maro.simulator.scenarios.supply_chain.objects import LeadingTimeInfo, SkuInfo, VendorLeadingTimeInfo
from maro.simulator.scenarios.supply_chain.sku_dynamics_sampler import (
    DataFileDemandSampler, OneTimeSkuPriceDemandSampler, SkuDynamicsSampler, SkuPriceInterface
)
from maro.simulator.scenarios.supply_chain.units import DistributionUnit, ProductUnit, StorageUnit
from maro.simulator.scenarios.supply_chain.units.distribution import DistributionUnitInfo
from maro.simulator.scenarios.supply_chain.units.product import ProductUnitInfo
from maro.simulator.scenarios.supply_chain.units.storage import StorageUnitInfo

if typing.TYPE_CHECKING:
    from maro.simulator.scenarios.supply_chain import UnitBase
    from maro.simulator.scenarios.supply_chain.datamodels.base import DataModelBase
    from maro.simulator.scenarios.supply_chain.world import World


# Mapping for supported sampler.
sampler_mapping = {
    "data": DataFileDemandSampler,
    "processed_price_demand": OneTimeSkuPriceDemandSampler,
}


@dataclass
class FacilityInfo:
    id: int
    name: str
    node_index: int
    class_name: type
    configs: dict
    skus: Dict[int, SkuInfo]
    upstream_vlt_infos: Dict[int, List[VendorLeadingTimeInfo]]
    downstreams: Dict[int, List[int]]  # Key: product_id; Value: facility id list
    storage_info: Optional[StorageUnitInfo]
    distribution_info: Optional[DistributionUnitInfo]
    products_info: Dict[int, ProductUnitInfo]  # Key: product_id


class FacilityBase(ABC):
    """Base of all facilities."""
    def __init__(
        self, id: int, name: str, data_model_name: str, data_model_index: int, world: World, config: dict,
    ) -> None:
        # Id and name of this facility.
        self.id: int = id
        self.name: str = name

        # Name of data model, and the facility instance's corresponding node index in snapshot list.
        self.data_model_name: str = data_model_name
        self.data_model_index: int = data_model_index

        # World of this facility belongs to.
        self.world: World = world

        # Configuration of this facility.
        self.configs: dict = config

        # SKUs in this facility.
        self.skus: Dict[int, SkuInfo] = {}
        self.sampler: Optional[SkuDynamicsSampler] = None

        # Product units for each sku in this facility.
        # Key is sku(product) id, value is the instance of product unit.
        self.products: Optional[Dict[int, ProductUnit]] = None

        # Storage unit in this facility.
        self.storage: Optional[StorageUnit] = None

        # Distribution unit in this facility.
        self.distribution: Optional[DistributionUnit] = None

        # Upstream facility vendor leading time infos.
        # Key: sku id;
        # Value: a list of vendor leading time info, including source facility, vehicle type, vlt, transportation cost.
        self.upstream_vlt_infos: Dict[int, List[VendorLeadingTimeInfo]] = defaultdict(list)

        # Downstream facility leading time infos.
        # Key: sku id;
        # Value: a list of leading time info, including destination facility, vehicle type, vlt, transportation cost.
        self.downstream_vlt_infos: Dict[int, List[LeadingTimeInfo]] = defaultdict(list)

        self.data_model: Optional[DataModelBase] = None

        # Children of this facility (valid units).
        self.children: List[UnitBase] = []

    def get_config(self, key: str, default: object = None) -> object:
        """Get specified configuration of facility.

        Args:
            key (str): Key of the configuration.
            default (object): Default value if key not exist, default is None.

        Returns:
            object: value in configuration.
        """
        return default if self.configs is None else self.configs.get(key, default)

    def initialize(self) -> None:
        """Initialize this facility after frame is ready."""
        self.data_model.initialize()

        # Put valid units into the children, used to simplify following usage.
        if self.storage is not None:
            self.children.append(self.storage)

        if self.distribution is not None:
            self.children.append(self.distribution)

        if self.products is not None:
            for product in self.products.values():
                self.children.append(product)

        sampler_cls_name = self.configs.get("dynamics_sampler_type", None)

        if sampler_cls_name is not None:
            assert sampler_cls_name in sampler_mapping
            sampler_cls = sampler_mapping[sampler_cls_name]

            assert issubclass(sampler_cls, SkuDynamicsSampler)
            self.sampler = sampler_cls(self.configs, self.world)

    def init_step(self, tick: int) -> None:
        """Update status before step. E.g. price updates, inventory updates, etc."""
        # Update SKU prices
        if self.sampler is not None and isinstance(self.sampler, SkuPriceInterface):
            for sku_id in self.skus.keys():
                price = self.sampler.sample_price(tick, sku_id)
                if price is not None:
                    self.skus[sku_id].price = price
                    # Update the corresponding property in data model.
                    self.products[sku_id].data_model.price = price

    def step(self, tick: int) -> None:
        """Push facility to next step.

        Args:
            tick (int): Current simulator tick.
        """
        for unit in self.children:
            unit.step(tick)

    def flush_states(self) -> None:
        """Flush states into frame."""
        for unit in self.children:
            unit.flush_states()

    def post_step(self, tick: int) -> None:
        """Post processing at the end of step."""
        for unit in self.children:
            unit.post_step(tick)

    def reset(self) -> None:
        """Reset facility for new episode."""
        for unit in self.children:
            unit.reset()

        if self.data_model is not None:
            self.data_model.reset()

    def get_in_transit_orders(self) -> Dict[int, int]:
        in_transit_orders = defaultdict(int)

        for product_id, product in self.products.items():
            if product.consumer is not None:
                in_transit_orders[product_id] = product.consumer.get_in_transit_quantity()

        return in_transit_orders

    def get_node_info(self) -> FacilityInfo:
        return FacilityInfo(
            id=self.id,
            name=self.name,
            node_index=self.data_model_index,
            class_name=type(self),
            configs=self.configs,
            skus=self.skus,
            upstream_vlt_infos=self.upstream_vlt_infos,
            downstreams={
                product_id: [info.dest_facility.id for info in info_list]
                for product_id, info_list in self.downstream_vlt_infos.items()
            },
            storage_info=self.storage.get_unit_info() if self.storage else None,
            distribution_info=self.distribution.get_unit_info() if self.distribution else None,
            products_info={
                product_id: product.get_unit_info()
                for product_id, product in self.products.items()
            },
        )

    def get_sku_cost(self, sku_id: int) -> float:
        # TODO: updating for manufacture, ...
        src_prices: List[float] = [
            vlt_info.src_facility.skus[sku_id].price
            for vlt_info in self.upstream_vlt_infos[sku_id]
        ]
        return np.mean(src_prices) if len(src_prices) > 0 else self.skus[sku_id].price

    def get_max_vlt(self, sku_id: int) -> int:
        max_vlt: int = 0
        for vlt_info in self.upstream_vlt_infos[sku_id]:
            max_vlt = max(max_vlt, vlt_info.vlt)
        return max_vlt
