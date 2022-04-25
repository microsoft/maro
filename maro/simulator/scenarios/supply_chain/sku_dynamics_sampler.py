# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import os.path
import typing
import warnings
from abc import ABCMeta, abstractmethod
from collections import defaultdict, namedtuple
from csv import DictReader
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Union

from dateutil.parser import parse
from tqdm import tqdm

from maro.data_lib.supply_chain import (
    DATE_INDEX_COLUMN_NAME, get_date_index, get_preprocessed_file_path, preprocess_file
)

if typing.TYPE_CHECKING:
    from maro.simulator.scenarios.supply_chain.world import World


@dataclass
class DynamicsInfoItem:
    column_name: str
    type_name: type
    default_value: object


class SkuDynamicsSampler(metaclass=ABCMeta):
    """Sampler to read SKU dynamics info from preprocessed data files, one store one file.
    The preprocessed file has a `DateIndex` column which represents its date's index, starting from 01/01/1970.

    NOTE:
        This sampler need to configure the start time that to be treated as tick 0 in world.settings, or
        it will use first row as start time.

    Args:
        configs (dict): Configuration of the facility it belongs to, it should contains following keys.
            . "file_path", the path to the data file.
            . "sku_column", column name contains sku name, must be in the facility's SKUs, or will be ignored.
            . "price_column", column name that will be treated as price.
            . "demand_column", column name that will be treated as demand from the end customers (sales).
            . "datetime_column", column name that contains datetime, NOTE: we will parse it that ignore the time zone.
        world (World): Current world this facility belongs to.
    """
    def __init__(self, configs: dict, world: World) -> None:
        self._configs: dict = configs
        self._world: World = world

        self._file_path = configs["file_path"]
        self._preprocessed_file_path = get_preprocessed_file_path(configs["file_path"])

        self._start_date_index: Optional[int] = None
        start_date_time = self._world.configs.settings["start_date_time"]
        if start_date_time is not None:
            self._start_date_index = get_date_index(parse(start_date_time))

        self._datetime_column_name = configs.get("datetime_column", "DT")
        self._sku_column_name = configs.get("sku_column", "SKU")
        self._info_dict: Dict[str, DynamicsInfoItem] = self._init_info_dict()

        # TODO: build up SC data pipeline, and use the processed files by default.
        if not os.path.exists(self._preprocessed_file_path):
            print(f"Preprocessed file {self._preprocessed_file_path} does not exist. Start preprocessing now.")
            preprocess_file(self._file_path, date_column_name=self._datetime_column_name)

        # TODO: add end_tick / max_tick parameter to close the file handler in time.
        self._cache = defaultdict(dict)

    @abstractmethod
    def _init_info_dict(self) -> Dict[str, DynamicsInfoItem]:
        raise NotImplementedError


class OneTimeSkuDynamicsSampler(SkuDynamicsSampler, metaclass=ABCMeta):
    """Load & cache all data when initializing."""
    def __init__(self, configs: dict, world: World) -> None:
        super(OneTimeSkuDynamicsSampler, self).__init__(configs, world)
        self._load_all_data()

    def _load_all_data(self) -> None:
        with open(self._preprocessed_file_path, "rt") as fp:
            reader = DictReader(fp)

            for row in tqdm(reader, desc=f"Loading data from {fp.name}"):
                date_index = int(row[DATE_INDEX_COLUMN_NAME])

                if self._start_date_index is None:
                    self._start_date_index = date_index

                # So one day one tick.
                target_tick = date_index - self._start_date_index

                sku_name = row[self._sku_column_name]
                # TODO: update to check with the facility's sku collections
                if sku_name not in self._world.sku_name2id_mapping:
                    continue
                sku_id = self._world.sku_name2id_mapping[sku_name]

                self._cache[target_tick][sku_id] = {}
                for attr_name, item in self._info_dict.items():
                    self._cache[target_tick][sku_id][attr_name] = item.type_name(row[item.column_name])


class StreamSkuDynamicsSampler(SkuDynamicsSampler, metaclass=ABCMeta):
    """Load & cache data in streaming fashion, i.e., only load data when necessary.

    `StreamSkuDynamicsSampler` works based on the following assumptions: the `tick` parameter of
    `sample_demand()` methods' calling is monotonically increasing, and the data file is also sorted by date
    increasingly.

    Using `StreamSkuDynamicsSampler` results in faster env creation (if we do not need to preprocess data)
    since we do not have to load all data at the beginning. But it also results in slower execution of the first
    episode because the data are loaded while executing the first episode. The execution efficiency will back to normal
    starting from the second episode.
    """
    def __init__(self, configs: dict, world: World) -> None:
        super(StreamSkuDynamicsSampler, self).__init__(configs, world)
        self._fp = open(self._preprocessed_file_path, "rt")
        self._reader = DictReader(self._fp)
        self._is_fp_closed = False
        self._latest_tick = None

    def _load_data_until_tick(self, tick: int) -> None:
        """Load all data that are no later than `tick`

        May load one more extra entry to ensure that all data at tick `tick` are loaded.
        """
        while not self._is_fp_closed and (self._latest_tick is None or self._latest_tick <= tick):
            row = next(self._reader)  # This entry may be after `tick`. We have to process it as we already loaded it.
            if row is None:
                self._fp.close()
                self._is_fp_closed = True
                break

            date_index = int(row[DATE_INDEX_COLUMN_NAME])

            if self._start_date_index is None:
                self._start_date_index = date_index

            # So one day one tick.
            self._latest_tick = target_tick = date_index - self._start_date_index

            sku_name = row[self._sku_column_name]
            if sku_name not in self._world.sku_name2id_mapping:
                continue
            sku_id = self._world.sku_name2id_mapping[sku_name]

            self._cache[target_tick][sku_id] = {}
            for attr_name, (column_name, type_name) in self._info_dict.items():
                self._cache[target_tick][sku_id][attr_name] = type_name(row[column_name])


class SkuPriceMixin(metaclass=ABCMeta):
    """Price sample interface."""

    @abstractmethod
    def sample_price(self, tick: int, product_id: int) -> Optional[float]:
        """Sample the price for specified product and tick.

        Args:
            tick (int): Tick of environment, NOTE: this tick is start from 0,
                you may need to transform it to your time system.
            product_id (int): Id of product to sample.

        Return:
            Optional[float]: the new price in the specific tick. None if not changed.
        """
        raise NotImplementedError


class SellerDemandMixin(metaclass=ABCMeta):
    """Demand sample interface, you can inherit from this to read from file or predict from a model."""

    @abstractmethod
    def sample_demand(self, tick: int, product_id: int) -> int:
        """Sample the demand for specified product and tick.

        Args:
            tick (int): Tick of environment, NOTE: this tick is start from 0,
                you may need to transform it to your time system.
            product_id (int): Id of product to sample.
        """
        raise NotImplementedError


class OneTimeSkuPriceDemandSampler(OneTimeSkuDynamicsSampler, SkuPriceMixin, SellerDemandMixin):
    def _init_info_dict(self) -> Dict[str, DynamicsInfoItem]:
        return {
            "Price": DynamicsInfoItem(self._configs.get("price_column", "Price"), float, None),
            "Demand": DynamicsInfoItem(self._configs.get("demand_column", "Demand"), int, 0),
        }

    def _sample_attr(self, tick: int, product_id: int, attr_name: str) -> object:
        if any([
            tick not in self._cache,
            product_id not in self._cache[tick],
            attr_name not in self._cache[tick][product_id],
        ]):
            return self._info_dict[attr_name].default_value

        return self._info_dict[attr_name].type_name(self._cache[tick][product_id][attr_name])

    def sample_price(self, tick: int, product_id: int) -> Optional[float]:
        price = self._sample_attr(tick, product_id, "Price")
        assert isinstance(price, float)
        return price

    def sample_demand(self, tick: int, product_id: int) -> int:
        demand = self._sample_attr(tick, product_id, "Demand")
        assert isinstance(demand, int)
        return demand


class StreamSkuPriceDemandSampler(StreamSkuDynamicsSampler, SkuPriceMixin, SellerDemandMixin):
    def _init_info_dict(self) -> Dict[str, DynamicsInfoItem]:
        return {
            "Price": DynamicsInfoItem(self._configs.get("price_column", "Price"), float, None),
            "Demand": DynamicsInfoItem(self._configs.get("demand_column", "Demand"), int, 0),
        }

    def _sample_attr(self, tick: int, product_id: int, attr_name: str) -> object:
        self._load_data_until_tick(tick)

        if any([
            tick not in self._cache,
            product_id not in self._cache[tick],
            attr_name not in self._cache[tick][product_id],
        ]):
            return self._info_dict[attr_name].default_value

        return self._info_dict[attr_name].type_name(self._cache[tick][product_id][attr_name])

    def sample_price(self, tick: int, product_id: int) -> Optional[float]:
        price = self._sample_attr(tick, product_id, "Price")
        assert isinstance(price, float)
        return price

    def sample_demand(self, tick: int, product_id: int) -> int:
        demand = self._sample_attr(tick, product_id, "Demand")
        assert isinstance(demand, int)
        return demand


SkuRow = namedtuple("SkuRow", ("price", "demand"))


class DataFileDemandSampler(SellerDemandMixin):
    """Sampler to read sample demand from data files, one store one file.

    NOTE:
        This sampler need to configure the start time that to be treated as tick 0 in world.settings, or
        it will use first row as start time.

    Args:
        configs (dict): Configuration from retail facility, it should contains following keys.
            . "file_path", the path to the data file
            . "sku_column", column name contains sku name, this must be match with current seller, or will be ignored.
            . "price_column", column name that will be treated as price.
            . "demand_column", column name that will be treated as sale number (demand).
            . "datetime_column", column name that contains datetime, NOTE: we will parse it that ignore the time zone.
    """
    def __init__(self, configs: dict, world: World) -> None:
        self._configs: dict = configs
        self._world: World = world

        self._file_path = configs["file_path"]

        # If start date time is None, then will use first row as start date time (tick 0).
        self._start_date_time: Optional[Union[str, datetime]] = self._world.configs.settings["start_date_time"]

        if self._start_date_time is not None:
            self._start_date_time = parse(self._start_date_time, ignoretz=True)

        self._sku_column_name = configs.get("sku_column", "SKU")
        self._price_column_name = configs.get("price_column", "Price")
        self._demand_column_name = configs.get("demand_column", "Sales")
        self._datetime_column_name = configs.get("datetime_column", "DT")

        # Tick -> sku -> (sale, price).
        self._cache = defaultdict(dict)
        self._cache_data()

    def sample_demand(self, tick: int, product_id: int) -> int:
        if tick not in self._cache or product_id not in self._cache[tick]:
            return 0

        return self._cache[tick][product_id].demand

    def _cache_data(self) -> None:
        with open(self._file_path, "rt") as fp:
            reader = DictReader(fp)

            for row in tqdm(reader, desc=f"Loading data from {fp.name}"):
                sku_name = row[self._sku_column_name]

                if sku_name not in self._world.sku_name2id_mapping:
                    continue

                demand = int(row[self._demand_column_name])
                price = float(row[self._price_column_name])
                date = parse(row[self._datetime_column_name], ignoretz=True)

                if self._start_date_time is None:
                    self._start_date_time = date

                # So one day one tick.
                target_tick = (date - self._start_date_time).days

                sku = self._world.get_sku_by_name(sku_name)

                if sku is not None:
                    self._cache[target_tick][sku.id] = SkuRow(price, demand)
                else:
                    warnings.warn(f"{sku_name} not configured in config file.")
