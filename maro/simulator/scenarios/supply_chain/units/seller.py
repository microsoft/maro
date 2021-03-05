
import numpy as np

from .base import UnitBase


class SellerUnit(UnitBase):
    """
    Unit that used to generate product consume demand, and move demand product from current storage.
    """
    def __init__(self):
        super(SellerUnit, self).__init__()

    def initialize(self, configs: dict):
        super(SellerUnit, self).initialize(configs)

    def step(self, tick: int):
        product_id = self.data.product_id
        sku = self.facility.sku_information[product_id]
        demand = self.market_demand()

        sold_qty = self.facility.storage.take_available(product_id, demand)

        self.data.total_sold += sold_qty
        self.data.sold = sold_qty
        self.data.demand = demand

    def post_step(self, tick: int):
        self.data.sold = 0
        self.data.demand = 0

    def reset(self):
        super(SellerUnit, self).reset()

    def market_demand(self):
        return int(np.random.gamma(self.data.sale_gamma))
