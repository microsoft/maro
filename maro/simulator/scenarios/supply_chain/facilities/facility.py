# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from __future__ import annotations

import collections
import typing
from abc import ABC
from collections import defaultdict
from typing import Dict, List, Optional

from maro.simulator.scenarios.supply_chain.objects import SkuInfo
from maro.simulator.scenarios.supply_chain.units import DistributionUnit, ProductUnit, StorageUnit

if typing.TYPE_CHECKING:
    from maro.simulator.scenarios.supply_chain.datamodels.base import DataModelBase
    from maro.simulator.scenarios.supply_chain.world import World
    from maro.simulator.scenarios.supply_chain import UnitBase


class FacilityBase(ABC):
    """Base of all facilities."""

    # Id of this facility.
    id: Optional[int] = None

    # Name of this facility.
    name: Optional[str] = None

    # World of this facility belongs to.
    world: Optional[World] = None

    # Skus in this facility.
    skus: Optional[Dict[int, SkuInfo]] = None

    # Product units for each sku in this facility.
    # Key is sku(product) id, value is the instance of product unit.
    products: Optional[Dict[int, ProductUnit]] = None

    # Storage unit in this facility.
    storage: Optional[StorageUnit] = None

    # Distribution unit in this facility.
    distribution: Optional[DistributionUnit] = None

    # Upstream facilities.
    # Key is sku id, value is the list of facilities from upstream.
    upstreams: Optional[Dict[int, List[FacilityBase]]] = None

    # Down stream facilities, value same as upstreams.
    downstreams: Optional[Dict[int, List[FacilityBase]]] = None

    # Configuration of this facility.
    configs: Optional[dict] = None

    # Name of data model, from configuration.
    data_model_name: Optional[str] = None

    # Index of the data model node.
    data_model_index: int = 0

    data_model: Optional[DataModelBase] = None

    # Children of this facility (valid units).
    children: Optional[list] = None

    # Facility's coordinates
    x: Optional[int] = None
    y: Optional[int] = None

    def __init__(self) -> None:
        self.upstreams = {}
        self.downstreams = collections.defaultdict(list)
        self.children: List[UnitBase] = []
        self.skus = {}

    def parse_skus(self, configs: dict) -> None:
        """Parse sku information from config.

        Args:
            configs (dict): Configuration of skus belongs to this facility.
        """
        for sku_name, sku_config in configs.items():
            sku_config['id'] = self.world.get_sku_id_by_name(sku_name)
            facility_sku = SkuInfo(**sku_config)
            self.skus[facility_sku.id] = facility_sku

    def parse_configs(self, configs: dict) -> None:
        """Parse configuration of this facility.

        Args:
            configs (dict): Configuration of this facility.
        """
        self.configs = configs

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

    def step(self, tick: int) -> None:
        """Push facility to next step.

        Args:
            tick (int): Current simulator tick.
        """
        for unit in self.children:
            unit.step(tick)
            unit.clear_action()

    # TODO: confirm Why not call flush_states() immediately after each unit.step()?
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

    def get_in_transit_orders(self) -> dict:
        in_transit_orders = defaultdict(int)

        for product_id, product in self.products.items():
            if product.consumer is not None:
                in_transit_orders[product_id] = product.consumer.get_in_transit_quantity()

        return in_transit_orders

    def set_action(self, action: object) -> None:
        pass

    def get_node_info(self) -> dict:
        products_info = {}

        for product_id, product in self.products.items():
            products_info[product_id] = product.get_unit_info()

        return {
            "id": self.id,
            "name": self.name,
            "class": type(self),
            "node_index": self.data_model_index,
            "units": {
                "storage": self.storage.get_unit_info() if self.storage is not None else None,
                "distribution": self.distribution.get_unit_info() if self.distribution is not None else None,
                "products": products_info,
            },
            "configs": self.configs,
            "skus": self.skus,
            "upstreams": {
                product_id: [f.id for f in source_list]
                for product_id, source_list in self.upstreams.items()
            },
            "downstreams": {
                product_id: [f.id for f in source_list]
                for product_id, source_list in self.downstreams.items()
            },
        }
