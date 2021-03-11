
from collections import defaultdict, Counter
from .base import UnitBase
from .order import Order


# TODO: we need another consumer type that read data from file or predict from model
class ConsumerUnit(UnitBase):
    """Unit that used to generate orders to purchase source materials from up stream facility.

    One consumer per sku.
    """
    def __init__(self):
        super(ConsumerUnit, self).__init__()

        # TODO: if we need this as state, we can add a new field to consumer
        # and fill the field in the post_step method.
        self.open_orders = defaultdict(Counter)

    def initialize(self, configs: dict):
        super(ConsumerUnit, self).initialize(configs)

        if len(self.data.sources) > 0:
            # we use 1st source as default one
            self.data.source_id = self.data.sources[0]

    def step(self, tick: int):
        # NOTE:
        # different with original code, we split the control into pieces,
        # and put in to frame, so the product_id always has value
        #
        data = self.data
        quantity = data.quantity

        if quantity <= 0 or len(data.sources) == 0:
            return

        source_id = data.source_id
        product_id = data.product_id

        vlt = data.vlt

        # NOTE:
        # we are using facility as source id, not the storage
        self.update_open_orders(source_id, product_id, quantity)

        order = Order(self.facility, product_id, quantity, vlt)

        source_facility = self.world.get_facility_by_id(source_id)

        data.order_product_cost = source_facility.distribution.place_order(order)

        data.total_purchased += quantity

        # update balance sheet
        data.balance_sheet_loss = -(data.order_product_cost + data.order_cost)

    def post_step(self, tick: int):
        super(ConsumerUnit, self).post_step(tick)

        data = self.data

        data.received = 0
        data.purchased = 0
        data.order_product_cost = 0

    def reset(self):
        super(ConsumerUnit, self).reset()

        self.open_orders.clear()

        if len(self.data.sources) > 0:
            # we use 1st source as default one
            self.data.source_id = self.data.sources[0]

    def set_action(self, action):
        # called before step
        data = self.data

        data.source_id = action.source_id
        data.quantity = action.quantity
        data.vlt = action.vlt

    def on_order_reception(self, source_id: int, product_id: int, quantity: int, original_quantity: int):
        self.data.total_received += quantity
        self.data.received += quantity

        self.update_open_orders(source_id, product_id, -original_quantity)

    def update_open_orders(self, source_id: int, product_id: int, qty_delta: int):
        if qty_delta > 0:
            # new order for product
            self.open_orders[source_id][product_id] += qty_delta
        else:
            # an order is completed, update the remaining number
            self.open_orders[source_id][product_id] += qty_delta

            # TODO: refine it later, seems like we do not need this
            # if len(self.open_orders[source_id]) == 0:
            #     del self.open_orders[source_id]

    def get_unit_info(self) -> dict:
        info = super(ConsumerUnit, self).get_unit_info()

        info["sku_id"] = self.data.product_id

        return info
