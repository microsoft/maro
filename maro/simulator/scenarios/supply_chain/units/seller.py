
import numpy as np
import random

from .base import UnitBase


class SellerUnit(UnitBase):
    """
    Unit that used to generate product consume demand, and move demand product from current storage.
    """
    def __init__(self):
        super(SellerUnit, self).__init__()

        self.gamma = 0

        self.demand_distribution = []

    def initialize(self, configs: dict, durations: int):
        super(SellerUnit, self).initialize(configs, durations)

        self.gamma = self.data.sale_gamma

        for _ in range(durations):
            self.demand_distribution.append(np.random.gamma(self.gamma))

    def step(self, tick: int):
        data = self.data

        product_id = data.product_id
        demand = self.market_demand(tick)

        sold_qty = self.facility.storage.take_available(product_id, demand)

        data.total_sold += sold_qty
        data.sold = sold_qty
        data.demand = demand

    def end_post_step(self, tick: int):
        # super(SellerUnit, self).post_step(tick)

        self.data.sold = 0
        self.data.demand = 0

    def reset(self):
        super(SellerUnit, self).reset()

    def market_demand(self, tick:int):
        return int(self.demand_distribution[tick])
