
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

        # attribution cache
        self.source_id = 0
        self.quantity = 0
        self.product_id = 0
        self.recieved = 0
        self.purchased = 0
        self.order_cost = 0


    def initialize(self, configs: dict, durations: int):
        if len(self.data.sources) > 0:
            # we use 1st source as default one
            self.source_id = self.data.sources[0]
            self.data.source_id = self.source_id

        self.product_id = self.data.product_id

    def step(self, tick: int):
        # NOTE:
        # different with original code, we split the control into pieces,
        # and put in to frame, so the product_id always has value
        #
        # id == 0 means invalid,as our id is 1 based
        if self.quantity <= 0 or self.source_id == 0:
            return

        # NOTE:
        # we are using facility as source id, not the storage
        self.update_open_orders(self.source_id, self.product_id, self.quantity)

        order = Order(self.facility, self.product_id, self.quantity, self.vlt)

        source_facility = self.world.get_facility_by_id(self.source_id)

        self.order_cost = source_facility.distribution.place_order(order)

        self.purchased = self.quantity

        # clear the action, as it should only be executed once.
        self.source_id = 0
        self.quantity = 0
        self.vlt = 0
    
    def begin_post_step(self, tick: int):
        if self.recieved > 0:
            self.data.received = self.recieved
            self.data.total_received += self.recieved

        if self.purchased > 0:
            self.data.purchased = self.purchased
            self.data.total_purchased += self.purchased

        if self.order_cost > 0:
            self.data.order_product_cost = self.order_cost

    def end_post_step(self, tick: int):
        if self.recieved > 0:
            self.data.received = 0
            self.recieved = 0

        if self.purchased > 0:
            self.data.purchased = 0
            self.purchased = 0

        if self.order_cost > 0:
            self.data.order_product_cost = 0
            self.order_cost = 0

    def reset(self):
        super(ConsumerUnit, self).reset()

        self.open_orders.clear()

        if len(self.data.sources) > 0:
            # we use 1st source as default one
            self.source_id = self.data.sources[0]
            self.data.source_id = self.source_id

    def set_action(self, action):
        # called before step
        self.source_id = action.source_id
        self.quantity = action.quantity
        self.vlt = action.vlt

        # record the action
        self.data.source_id = action.source_id
        self.data.quantity = action.quantity
        self.data.vlt = action.vlt

    def on_order_reception(self, source_id: int, product_id: int, quantity: int, original_quantity: int):
        self.recieved += quantity
        # self.data.total_received += quantity
        # self.data.received += quantity

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

        info["sku_id"] = self.product_id

        return info
