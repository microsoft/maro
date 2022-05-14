# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from maro.simulator.scenarios.supply_chain.datamodels import SellerDataModel
from maro.simulator.scenarios.supply_chain.sku_dynamics_sampler import SellerDemandMixin

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

        sku = self.facility.skus[self.sku_id]

        self._gamma = sku.sale_gamma

        assert isinstance(self.data_model, SellerDataModel)
        self.data_model.initialize(sku.backlog_ratio)

        self._sale_hist = [self._gamma] * self.config["sale_hist_len"]
        self._demand_hist = [self._gamma] * self.config["sale_hist_len"]

    def pre_step(self, tick: int) -> None:
        if self._sold > 0:
            self.data_model.sold = 0
            self._sold = 0

        if self._demand > 0:
            self.data_model.demand = 0
            self._demand = 0

    def step(self, tick: int) -> None:
        demand = self.market_demand(tick)

        # What seller does is just count down the product number.
        sold_qty = self.facility.storage.take_available(self.sku_id, demand)

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

    def reset(self) -> None:
        super(SellerUnit, self).reset()

        # Reset status in Python side.
        self._sold = 0
        self._demand = 0
        self._total_sold = 0
        self._total_demand = 0

        self._sale_hist = [self._gamma] * self.config["sale_hist_len"]
        self._demand_hist = [self._gamma] * self.config["sale_hist_len"]

    def sale_mean(self) -> float:  # TODO: renamed to median demand
        return float(np.median(self._demand_hist))

    def sale_std(self) -> float:
        return float(np.std(self._sale_hist))

    def get_node_info(self) -> SellerUnitInfo:
        return SellerUnitInfo(
            **super(SellerUnit, self).get_unit_info().__dict__,
        )


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
    sampler: SellerDemandMixin = None

    def market_demand(self, tick: int) -> int:
        return self.sampler.sample_demand(tick, self.sku_id)
