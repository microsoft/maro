
from collections import namedtuple
from typing import List
from .base import FacilityBase


class WarehouseFacility(FacilityBase):
    SkuInfo = namedtuple("SkuInfo", ("name", "init_in_stock", "id", "price", "delay_order_penalty"))

    # storage unit
    storage = None

    # distribution unit
    distribution = None

    # vehicle list
    transports = None

    # consumers that will generate order to purchase productions
    # one sku one consumer
    consumers = None

    def step(self, tick: int):
        self.storage.step(tick)
        self.distribution.step(tick)

        for transport in self.transports:
            transport.step(tick)

        for consumer in self.consumers.values():
            consumer.step(tick)

    def build(self, configs: dict):
        self.configs = configs

        # construct storage
        self.storage = self.world.build_unit(configs["storage"]["class"])
        self.storage.data_class = configs["storage"]["data"]["class"]

        self.storage.world = self.world
        self.storage.facility = self
        self.storage.data_index = self.world.register_data_class(self.storage.id, self.storage.data_class)

        # construct transport
        self.transports = []

        for facility_conf in configs["transports"]:
            transport = self.world.build_unit(facility_conf["class"])
            transport.data_class = facility_conf["data"]["class"]

            transport.world = self.world
            transport.facility = self
            transport.data_index = self.world.register_data_class(transport.id, transport.data_class)

            self.transports.append(transport)

        # construct distribution
        self.distribution = self.world.build_unit(configs["distribution"]["class"])
        self.distribution.data_class = configs["distribution"]["data"]["class"]

        self.distribution.world = self.world
        self.distribution.facility = self
        self.distribution.data_index = self.world.register_data_class(self.distribution.id, self.distribution.data_class)

        # sku information
        self.sku_information = {}
        self.consumers = {}

        for sku_name, sku_config in configs["skus"].items():
            sku = self.world.get_sku(sku_name)
            sku_info = WarehouseFacility.SkuInfo(
                sku_name,
                sku_config["init_stock"],
                sku.id,
                sku_config.get("price", 0),
                sku_config.get("delay_order_penalty", 0)
            )

            self.sku_information[sku.id] = sku_info

            consumer = self.world.build_unit(configs["consumers"]["class"])
            consumer.data_class = configs["consumers"]["data"]["class"]

            consumer.world = self.world
            consumer.facility = self
            consumer.data_index = self.world.register_data_class(consumer.id, consumer.data_class)

            self.consumers[sku.id] = consumer

    def initialize(self):
        # init components that related with sku number
        self._init_by_skus()

        # called after build, here we have the data model, we can initialize them.
        self.storage.initialize(self.configs.get("storage", {}))
        self.distribution.initialize(self.configs.get("distribution", {}))

        transports_conf = self.configs.get("transports", [])

        for index, transport in enumerate(self.transports):
            transport.initialize(transports_conf[index])

        for sku_id, consumer in self.consumers.items():
            consumer.initialize({
                "data": {
                    "order_cost": self.configs.get("order_cost", 0),
                    "consumer_product_id": sku_id
                }
            })

    def reset(self):
        # NOTE: as we are using list attribute now, theirs size will be reset to defined one after frame.reset,
        # so we have to init them again.
        self._init_by_skus()

        self.storage.reset()
        self.distribution.reset()

        for vehicle in self.transports:
            vehicle.reset()

        for consumer in self.consumers.values():
            consumer.reset()

    def post_step(self, tick: int):
        self.storage.post_step(tick)
        self.distribution.post_step(tick)

        for vehicle in self.transports:
            vehicle.post_step(tick)

        for consumer in self.consumers.values():
            consumer.post_step(tick)

    def get_node_info(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "class": type(self),
            "units": {
                "storage": self.storage.get_unit_info(),
                "consumers": [consumer.get_unit_info() for consumer in self.consumers.values()],
                "transports": [vehicle.get_unit_info() for vehicle in self.transports],
                "distribution": self.distribution.get_unit_info()
            }
        }

    def _init_by_skus(self):
        for _, sku in self.sku_information.items():
            # update storage's production info
            self.storage.data.product_list.append(sku.id)
            self.storage.data.product_number.append(sku.init_in_stock)

            # update distribution's production info
            self.distribution.data.product_list.append(sku.id)
            self.distribution.data.check_in_price.append(0)
            self.distribution.data.delay_order_penalty.append(0)

        if self.upstreams is not None:
            # update the source facilities for each consumer
            for sku_id, source_facilities in self.upstreams.items():
                consumer = self.consumers[sku_id]

                for facility_id in source_facilities:
                    consumer.data.sources.append(facility_id)
