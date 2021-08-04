# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import warnings
from abc import ABC, abstractmethod
from collections import namedtuple
from csv import DictReader
from datetime import datetime
from typing import Optional, Union

from dateutil import parser

from .seller import SellerUnit


class SellerDemandSampler(ABC):
    """Base class of seller unit demand sampler, you can inherit from this to read from file or predict from a model.

    Args:
        configs (dict): Configuration from retailer facility, contains keys for a special sampler.
        world (World): Current world this retail belongs to.
    """

    def __init__(self, configs: dict, world):
        self._configs = configs
        self._world = world

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

    def __init__(self, configs: dict, world):
        super(DataFileDemandSampler, self).__init__(configs, world)

        self._file_path = configs["file_path"]

        # If start date time is None, then will use first row as start date time (tick 0).
        self._start_date_time: Optional[Union[str, datetime]] = self._world.configs.settings["start_date_time"]

        if self._start_date_time is not None:
            self._start_date_time = parser.parse(self._start_date_time, ignoretz=True)

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

    def _cache_data(self):
        with open(self._file_path, "rt") as fp:
            reader = DictReader(fp)

            for row in reader:
                sku_name = row[self._sku_column_name]

                sales = int(row[self._sale_column_name])
                price = float(row[self._price_column_name])
                date = parser.parse(row[self._datetime_column_name], ignoretz=True)

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

    # Sample used to sample demand.
    sampler: SellerDemandSampler = None

    def market_demand(self, tick: int) -> int:
        return self.sampler.sample_demand(self.product_id, tick)
