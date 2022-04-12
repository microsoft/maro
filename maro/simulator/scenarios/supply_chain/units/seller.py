# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import typing
import numpy as np
import warnings
from abc import ABC, abstractmethod
from collections import namedtuple
from csv import DictReader
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Union

import dateutil

from maro.simulator.scenarios.supply_chain.datamodels import SellerDataModel

from .extendunitbase import ExtendUnitBase, ExtendUnitInfo
from .unitbase import UnitBase

if typing.TYPE_CHECKING:
    from maro.simulator.scenarios.supply_chain.facilities import FacilityBase
    from maro.simulator.scenarios.supply_chain.world import World


@dataclass
class SellerUnitInfo(ExtendUnitInfo):
    pass


class SellerUnit(ExtendUnitBase):
    """
    Unit that used to generate product consume demand, and move demand product from current storage.
    """
    def __init__(
        self, id: int, data_model_name: Optional[str], data_model_index: Optional[int],
        facility: FacilityBase, parent: Union[FacilityBase, UnitBase], world: World, config: dict
    ) -> None:
        super(SellerUnit, self).__init__(
            id, data_model_name, data_model_index, facility, parent, world, config
        )

        self._gamma = 0

        # Attribute cache.
        self._sold = 0
        self._demand = 0
        self._total_sold = 0
        self._total_demand = 0

        self._sale_hist = []
        self._demand_hist = []

    def market_demand(self, tick: int) -> int:
        """Generate market demand for current tick.

        Args:
            tick (int): Current simulator tick.

        Returns:
            int: Demand number.
        """
        return int(np.random.gamma(self._gamma))

    def initialize(self) -> None:
        super(SellerUnit, self).initialize()

        sku = self.facility.skus[self.product_id]

        self._gamma = sku.sale_gamma

        assert isinstance(self.data_model, SellerDataModel)
        self.data_model.initialize(sku.price, sku.backlog_ratio)

        self._sale_hist = [self._gamma] * self.config["sale_hist_len"]
        self._demand_hist = [self._gamma] * self.config["sale_hist_len"]

    def _step_impl(self, tick: int) -> None:
        demand = self.market_demand(tick)

        # What seller does is just count down the product number.
        sold_qty = self.facility.storage.take_available(self.product_id, demand)

        self._total_sold += sold_qty
        self._sold = sold_qty
        self._demand = demand
        self._total_demand += demand

        self._sale_hist.append(sold_qty)
        self._sale_hist = self._sale_hist[1:]
        self._demand_hist.append(demand)
        self._demand_hist = self._demand_hist[1:]

    def flush_states(self) -> None:
        if self._sold > 0:
            self.data_model.sold = self._sold
            self.data_model.total_sold = self._total_sold

        if self._demand > 0:
            self.data_model.demand = self._demand
            self.data_model.total_demand = self._total_demand

    def post_step(self, tick: int) -> None:
        super(SellerUnit, self).post_step(tick)

        if self._sold > 0:
            self.data_model.sold = 0
            self._sold = 0

        if self._demand > 0:
            self.data_model.demand = 0
            self._demand = 0

    def reset(self):
        super(SellerUnit, self).reset()

        # Reset status in Python side.
        self._sold = 0
        self._demand = 0
        self._total_sold = 0
        self._total_demand = 0

        self._sale_hist = [self._gamma] * self.config["sale_hist_len"]
        self._demand_hist = [self._gamma] * self.config["sale_hist_len"]

    def sale_mean(self) -> float:
        return float(np.median(self._demand_hist))

    def sale_std(self) -> float:
        return float(np.std(self._sale_hist))

    def get_node_info(self) -> SellerUnitInfo:
        return SellerUnitInfo(
            **super(SellerUnit, self).get_unit_info().__dict__,
        )


class SellerDemandSampler(ABC):
    """Base class of seller unit demand sampler, you can inherit from this to read from file or predict from a model.

    Args:
        configs (dict): Configuration from retailer facility, contains keys for a special sampler.
        world (World): Current world this retail belongs to.
    """

    def __init__(self, configs: dict, world: World) -> None:
        self._configs: dict = configs
        self._world: World = world

    @abstractmethod
    def sample_demand(self, product_id: int, tick: int) -> int:
        """Sample the demand for specified product and tick.

        Args:
            product_id (int): Id of product to sample.
            tick (int): Tick of environment, NOTE: this tick is start from 0,
                you may need to transform it to your time system.
        """
        pass


class DataFileDemandSampler(SellerDemandSampler):
    """Sampler to read sample demand from data files, one store one file.

    NOTE:
        This sampler need to configure the start time that to be treated as tick 0 in world.settings, or
        it will use first row as start time.

    Args:
        configs (dict): Configuration from retail facility, it should contains following keys.
            . "file_path", the path to the data file
            . "sku_column", column name contains sku name, this must be match with current seller, or will be ignored.
            . "price_column", column name that will be treated as price.
            . "sale_column", column name that will be treated as sale number (demand).
            . "datetime_column", column name that contains datetime, NOTE: we will parse it that ignore the time zone.
    """

    SkuRow = namedtuple("SkuRow", ("price", "sales"))

    def __init__(self, configs: dict, world: World) -> None:
        super(DataFileDemandSampler, self).__init__(configs, world)

        self._file_path = configs["file_path"]

        # If start date time is None, then will use first row as start date time (tick 0).
        self._start_date_time: Optional[Union[str, datetime]] = self._world.configs.settings["start_date_time"]

        if self._start_date_time is not None:
            self._start_date_time = dateutil.parser.parse(self._start_date_time, ignoretz=True)

        self._sku_column_name = configs.get("sku_column", "SKU")
        self._price_column_name = configs.get("price_column", "Price")
        self._sale_column_name = configs.get("sale_column", "Sales")
        self._datetime_column_name = configs.get("datetime_column", "DT")

        # Tick -> sku -> (sale, price).
        self._cache = {}

        self._cache_data()

    def sample_demand(self, product_id: int, tick: int) -> int:
        if tick not in self._cache or product_id not in self._cache[tick]:
            return 0

        return self._cache[tick][product_id].sales

    def _cache_data(self) -> None:
        with open(self._file_path, "rt") as fp:
            reader = DictReader(fp)

            for row in reader:
                sku_name = row[self._sku_column_name]

                if sku_name not in self._world._sku_name2id_mapping:
                    continue

                sales = int(row[self._sale_column_name])
                price = float(row[self._price_column_name])
                date = dateutil.parser.parse(row[self._datetime_column_name], ignoretz=True)

                if self._start_date_time is None:
                    self._start_date_time = date

                # So one day one tick.
                target_tick = (date - self._start_date_time).days

                if target_tick not in self._cache:
                    self._cache[target_tick] = {}

                sku = self._world.get_sku_by_name(sku_name)

                if sku is not None:
                    self._cache[target_tick][sku.id] = DataFileDemandSampler.SkuRow(price, sales)
                else:
                    warnings.warn(f"{sku_name} not configured in config file.")


class OuterSellerUnit(SellerUnit):
    """Seller that demand is from out side sampler, like a data file or data model prediction."""

    def __init__(
        self, id: int, data_model_name: Optional[str], data_model_index: Optional[int],
        facility: FacilityBase, parent: Union[FacilityBase, UnitBase], world: World, config: dict
    ) -> None:
        super(OuterSellerUnit, self).__init__(
            id, data_model_name, data_model_index, facility, parent, world, config
        )

    # Sample used to sample demand.
    sampler: SellerDemandSampler = None

    def market_demand(self, tick: int) -> int:
        return self.sampler.sample_demand(self.product_id, tick)
