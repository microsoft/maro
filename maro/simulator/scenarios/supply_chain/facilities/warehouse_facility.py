
from collections import namedtuple
from typing import List
from .base import FacilityBase


class WarehouseFacility(FacilityBase):
    SkuInfo = namedtuple("SkuInfo", ("name", "init_in_stock", "id"))

    # storage unit
    storage = None

    # distribution unit
    distribution = None

    # vehicle list
    transports = None

    def step(self, tick: int):
        self.storage.step(tick)
        self.distribution.step(tick)

        for transport in self.transports:
            transport.step(tick)

    def build(self, configs: dict):
        self.configs = configs

        # TODO: following strings should from config later

        # construct storage
        self.storage = self.world.build_unit("StorageUnit")
        self.storage.data_class = "StorageDataModel"

        self.storage.world = self.world
        self.storage.facility = self
        self.storage.data_index = self.world.register_data_class(self.storage.data_class)

        # construct transport
        self.transports = []

        for facility_conf in configs["transports"]:
            transport = self.world.build_unit("TransportUnit")
            transport.data_class = "TransportDataModel"

            transport.world = self.world
            transport.facility = self
            transport.data_index = self.world.register_data_class(transport.data_class)

            self.transports.append(transport)

        # construct distribution
        self.distribution = self.world.build_unit("DistributionUnit")
        self.distribution.data_class = "DistributionDataModel"

        self.distribution.world = self.world
        self.distribution.facility = self
        self.distribution.data_index = self.world.register_data_class(self.distribution.data_class)

        # sku information
        self.sku_information = {}

        for sku_name, sku_config in configs["skus"].items():
            sku = self.world.get_sku(sku_name)
            sku_info = WarehouseFacility.SkuInfo(sku_name, sku_config["init_stock"], sku.id)

            self.sku_information[sku.id] = sku_info

    def initialize(self):
        # init components that related with sku number
        self._init_by_skus()

        # called after build, here we have the data model, we can initialize them.
        self.storage.initialize(self.configs.get("storage", {}))
        self.distribution.initialize(self.configs.get("distribution", {}))

        transports_conf = self.configs["transports"]

        for index, transport in enumerate(self.transports):
            transport.initialize(transports_conf[index])

    def reset(self):
        self.storage.reset()
        self.distribution.reset()

        for vehicle in self.transports:
            vehicle.reset()

        # NOTE: as we are using list attribute now, theirs size will be reset to defined one after frame.reset,
        # so we have to init them again.
        self._init_by_skus()

    def post_step(self, tick: int):
        self.storage.post_step(tick)
        self.distribution.post_step(tick)

        for vehicle in self.transports:
            vehicle.post_step(tick)

    def _init_by_skus(self):
        for _, sku in self.sku_information.items():
            # update storage's production info
            self.storage.data.product_list.append(sku.id)
            self.storage.data.product_number.append(sku.init_in_stock)

            # update distribution's production info
            self.distribution.data.product_list.append(sku.id)
            self.distribution.data.check_in_price.append(0)
            self.distribution.data.delay_order_penalty.append(0)
