# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import typing
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

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
        facility: FacilityBase, parent: Union[FacilityBase, UnitBase], world: World, config: dict,
    ) -> None:
        super(SellerUnit, self).__init__(
            id, data_model_name, data_model_index, facility, parent, world, config,
        )

        self._gamma = 0

        # Attribute cache.
        self._sold = 0
        self._demand = 0
        self._total_sold = 0
        self._total_demand = 0

        self._sale_hist = []

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
        self.data_model.initialize(sku.backlog_ratio)

        self._sale_hist = [self._gamma] * self.config["sale_hist_len"]

    def _step_impl(self, tick: int) -> None:
        demand = self.market_demand(tick)

        # What seller does is just count down the product number.
        sold_qty = self.facility.storage.take_available(self.product_id, demand)

        self._total_sold += sold_qty
        self._sold = sold_qty
        self._demand = demand
        self._total_demand += demand

        self._sale_hist.append(demand)
        self._sale_hist = self._sale_hist[1:]

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

    def reset(self) -> None:
        super(SellerUnit, self).reset()

        # Reset status in Python side.
        self._sold = 0
        self._demand = 0
        self._total_sold = 0
        self._total_demand = 0

        self._sale_hist = [self._gamma] * self.config["sale_hist_len"]

    def sale_mean(self) -> float:
        return float(np.mean(self._sale_hist))

    def sale_std(self) -> float:
        return float(np.std(self._sale_hist))

    def get_node_info(self) -> SellerUnitInfo:
        return SellerUnitInfo(
            **super(SellerUnit, self).get_unit_info().__dict__,
        )


class SellerDemandInterface(metaclass=ABCMeta):
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


class OuterSellerUnit(SellerUnit):
    """Seller that demand is from out side sampler, like a data file or data model prediction."""

    def __init__(
        self, id: int, data_model_name: Optional[str], data_model_index: Optional[int],
        facility: FacilityBase, parent: Union[FacilityBase, UnitBase], world: World, config: dict,
    ) -> None:
        super(OuterSellerUnit, self).__init__(
            id, data_model_name, data_model_index, facility, parent, world, config,
        )

    # Sample used to sample demand.
    sampler: SellerDemandInterface = None

    def market_demand(self, tick: int) -> int:
        return self.sampler.sample_demand(tick, self.product_id)
