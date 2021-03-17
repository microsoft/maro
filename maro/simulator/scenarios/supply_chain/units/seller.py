# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import numpy as np

from .skuunit import SkuUnit


class SellerUnit(SkuUnit):
    """
    Unit that used to generate product consume demand, and move demand product from current storage.
    """

    def __init__(self):
        super(SellerUnit, self).__init__()

        self.gamma = 0
        self.durations = 0
        self.demand_distribution = []

        # Attribute cache.
        self.sold = 0
        self.demand = 0
        self.total_sold = 0
        self.product_id = 0

    def market_demand(self, tick: int) -> int:
        """Generate market demand for current tick.

        Args:
            tick (int): Current simulator tick.

        Returns:
            int: Demand number.
        """
        return int(self.demand_distribution[tick])

    def initialize(self):
        super(SellerUnit, self).initialize()

        unit_price = self.config.get("unit_price", 0)
        self.gamma = self.config.get("sale_gamma", 0)
        backlog_ratio = self.config.get("backlog_ratio", 1)

        self.data_model.initialize(
            unit_price=unit_price,
            sale_gamma=self.gamma,
            product_id=self.product_id,
            backlog_ratio=backlog_ratio
        )

        self.durations = self.world.durations

        for _ in range(self.durations):
            self.demand_distribution.append(np.random.gamma(self.gamma))

    def step(self, tick: int):
        demand = self.market_demand(tick)

        # what seller does is just count down the product number.
        sold_qty = self.facility.storage.take_available(self.product_id, demand)

        self.total_sold += sold_qty
        self.sold = sold_qty
        self.demand = demand

    def flush_states(self):
        self.data_model.sold = self.sold
        self.data_model.demand = self.demand
        self.data_model.total_sold = self.total_sold

    def post_step(self, tick: int):
        super(SellerUnit, self).post_step(tick)

        if self.sold > 0:
            self.data_model.sold = 0

        if self.demand > 0:
            self.data_model.demand = 0

    def reset(self):
        super(SellerUnit, self).reset()

        # TODO: regenerate the demand distribution?
        # self.demand_distribution.clear()

        # for _ in range(self.durations):
        #     self.demand_distribution.append(np.random.gamma(self.gamma))
