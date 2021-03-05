
from collections import namedtuple
from .base import FacilityBase


class SupplierFacility(FacilityBase):
    SkuInfo = namedtuple("SkuInfo", ("name", "id", "init_in_stock", "production_rate", "type", "cost"))

    storage = None
    distribution = None
    transports = None
    suppliers = None
    consumers = None

    def step(self, tick: int):
        self.storage.step(tick)
        self.distribution.step(tick)

        for vehicle in self.transports:
            vehicle.step(tick)

        for supplier in self.suppliers.values():
            supplier.step(tick)

        for consumer in self.consumers.values():
            consumer.step(tick)

    def build(self, configs: dict):
        self.configs = configs

        # TODO: dup code from facilities, refactoring later

        # construct storage
        self.storage = self.world.build_unit("StorageUnit")
        self.storage.data_class = "StorageDataModel"

        self.storage.world = self.world
        self.storage.facility = self
        self.storage.data_index = self.world.register_data_class(self.storage.id, self.storage.data_class)

        # construct transport
        self.transports = []

        for _ in configs["transports"]:
            transport = self.world.build_unit("TransportUnit")
            transport.data_class = "TransportDataModel"

            transport.world = self.world
            transport.facility = self
            transport.data_index = self.world.register_data_class(transport.id, transport.data_class)

            self.transports.append(transport)

        # construct distribution
        self.distribution = self.world.build_unit("DistributionUnit")
        self.distribution.data_class = "DistributionDataModel"

        self.distribution.world = self.world
        self.distribution.facility = self
        self.distribution.data_index = self.world.register_data_class(self.distribution.id, self.distribution.data_class)

        # sku information
        self.sku_information = {}
        self.suppliers = {}
        self.consumers = {}

        for sku_name, sku_config in configs["skus"].items():
            sku = self.world.get_sku(sku_name)
            sku_info = SupplierFacility.SkuInfo(
                sku_name,
                sku.id,
                sku_config["init_in_stock"],
                sku_config.get("production_rate", 0),
                sku_config["type"],
                sku_config.get("cost", 0)
            )

            self.sku_information[sku.id] = sku_info

            # TODO: make it an enum later.
            if sku_info.type == "production":
                # one supplier per sku
                supplier = self.world.build_unit("ManufacturingUnit")
                supplier.data_class = "ManufactureDataModel"

                supplier.world = self.world
                supplier.facility = self
                supplier.data_index = self.world.register_data_class(supplier.id, supplier.data_class)

                self.suppliers[sku.id] = supplier
            else:
                consumer = self.world.build_unit("ConsumerUnit")
                consumer.data_class = "ConsumerDataModel"

                consumer.world = self.world
                consumer.facility = self
                consumer.data_index = self.world.register_data_class(consumer.id, consumer.data_class)

                self.consumers[sku.id] = consumer

    def initialize(self):
        # DO init by skus first, as other components may depend on sku information
        self._init_by_skus()

        for _, sku in self.sku_information.items():
            if sku.id in self.suppliers:
                supplier = self.suppliers[sku.id]

                # build parameters to initialize the data model
                supplier.initialize({
                    "data": {
                        "production_rate": sku.production_rate,
                        "output_product_id": sku.id,
                        "product_unit_cost": sku.cost
                    }
                })

        for consumer in self.consumers.values():
            consumer.initialize({
                "data": {
                    # TODO: move to config
                    "order_cost": self.configs.get("order_cost", 0)
                }
            })

        self.storage.initialize(self.configs.get("storage", {}))
        self.distribution.initialize(self.configs.get("distribution", {}))

        transports_conf = self.configs["transports"]

        for index, transport in enumerate(self.transports):
            transport.initialize(transports_conf[index])

    def post_step(self, tick: int):
        self.storage.post_step(tick)
        self.distribution.post_step(tick)

        for vehicle in self.transports:
            vehicle.post_step(tick)

        for supplier in self.suppliers.values():
            supplier.post_step(tick)

        for consumer in self.consumers.values():
            consumer.post_step(tick)

    def reset(self):
        self.storage.reset()
        self.distribution.reset()

        for vehicle in self.transports:
            vehicle.reset()

        for supplier in self.suppliers.values():
            supplier.reset()

        for consumer in self.consumers.values():
            consumer.reset()

        self._init_by_skus()

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
