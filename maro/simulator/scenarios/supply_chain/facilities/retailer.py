
from collections import namedtuple

from .base import FacilityBase


class RetailerFacility(FacilityBase):
    SkuInfo = namedtuple("SkuInfo", ("name", "id", "price", "cost", "init_in_stock", "sale_gamma"))

    storage = None
    consumers: dict = None
    sellers: dict = None

    def step(self, tick: int):
        self.storage.step(tick)

        self.data.balance_sheet_profit += self.storage.data.balance_sheet_profit
        self.data.balance_sheet_loss += self.storage.data.balance_sheet_loss

        if self.consumers is not None:
            for consumer in self.consumers.values():
                consumer.step(tick)

                self.data.balance_sheet_profit += consumer.data.balance_sheet_profit
                self.data.balance_sheet_loss += consumer.data.balance_sheet_loss

        if self.sellers is not None:
            for seller in self.sellers.values():
                seller.step(tick)

                self.data.balance_sheet_profit += seller.data.balance_sheet_profit
                self.data.balance_sheet_loss += seller.data.balance_sheet_loss

    def post_step(self, tick: int):
        self.storage.post_step(tick)

        if self.consumers is not None:
            for consumer in self.consumers.values():
                consumer.post_step(tick)

        if self.sellers is not None:
            for seller in self.sellers.values():
                seller.post_step(tick)

        self.data.balance_sheet_profit = 0
        self.data.balance_sheet_loss = 0

    def build(self, configs: dict):
        self.configs = configs

        # construct storage
        self.storage = self.world.build_unit(configs["storage"]["class"])
        self.storage.data_class = configs["storage"]["data"]["class"]

        self.storage.world = self.world
        self.storage.facility = self
        self.storage.data_index = self.world.register_data_class(self.storage.id, self.storage.data_class)

        self.sku_information = {}
        self.consumers = {}
        self.sellers = {}

        for sku_name, sku_config in configs["skus"].items():
            sku = self.world.get_sku(sku_name)
            sku_info = RetailerFacility.SkuInfo(
                sku_name,
                sku.id,
                sku_config.get("price", 0),
                sku_config.get("cost", 0),
                sku_config["init_in_stock"],
                sku_config["sale_gamma"]
            )

            self.sku_information[sku.id] = sku_info

            # all sku in retail are final production for sale, no material
            consumer = self.world.build_unit(configs["consumers"]["class"])
            consumer.data_class = configs["consumers"]["data"]["class"]

            consumer.world = self.world
            consumer.facility = self
            consumer.data_index = self.world.register_data_class(consumer.id, consumer.data_class)

            self.consumers[sku.id] = consumer

            # seller for this sku
            seller = self.world.build_unit(configs["sellers"]["class"])
            seller.data_class = configs["sellers"]["data"]["class"]

            seller.world = self.world
            seller.facility = self
            seller.data_index = self.world.register_data_class(seller.id, seller.data_class)

            self.sellers[sku.id] = seller

    def initialize(self):
        self.data.set_id(self.id, self.id)
        self.data.initialize({})

        self.storage.prepare_data()

        # prepare data for units
        for consumer in self.consumers.values():
            consumer.prepare_data()

        for seller in self.sellers.values():
            seller.prepare_data()

        self._init_by_sku()

        self.storage.initialize(self.configs.get("storage", {}))

        for sku_id, consumer in self.consumers.items():
            consumer.initialize({
                "data": {
                    # TODO: move to config
                    "order_cost": self.configs.get("order_cost", 0),
                    "consumer_product_id": sku_id
                }
            })

        for sku_id, seller in self.sellers.items():
            sku = self.sku_information[sku_id]

            seller.initialize({
                "data": {
                    "unit_price": sku.price,
                    "sale_gamma": sku.sale_gamma,
                    "product_id": sku_id
                }
            })

    def reset(self):
        self._init_by_sku()

        self.storage.reset()

        if self.consumers is not None:
            for consumer in self.consumers.values():
                consumer.reset()

        if self.sellers is not None:
            for seller in self.sellers.values():
                seller.reset()

    def get_node_info(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "class": type(self),
            "units": {
                "storage": self.storage.get_unit_info(),
                "consumers": [consumer.get_unit_info() for consumer in self.consumers.values()],
                "sellers": [seller.get_unit_info() for seller in self.sellers.values()]
            }
        }

    def _init_by_sku(self):
        for _, sku in self.sku_information.items():
            # update storage's production info
            self.storage.data.product_list.append(sku.id)
            self.storage.data.product_number.append(sku.init_in_stock)

        if self.upstreams is not None:
            # update the source facilities for each consumer
            for sku_id, source_facilities in self.upstreams.items():
                consumer = self.consumers[sku_id]

                for facility_id in source_facilities:
                    consumer.data.sources.append(facility_id)
