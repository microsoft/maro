
from typing import List


class WarehouseFacility:
    world = None

    storage = None
    distribution = None
    transports = None

    configs: dict = None

    id: int = None

    def __init__(self):
        pass

    def step(self, tick: int):

        self.storage.step(tick)
        self.distribution.step(tick)

        for transport in self.transports:
            transport.step(tick)

    def build(self, configs: dict):
        self.configs = configs

        # TODO: from config later
        self.storage = self.world.build_logic("StorageLogic")
        self.storage.data_class = "StorageDataModel"

        self.storage.world = self.world
        self.storage.facility = self
        self.storage.data_index = self.world.register_datamodel(self.storage.data_class)

        # construct transport
        self.transports = []

        for facility_conf in configs["transports"]:
            transport = self.world.build_logic("TransportLogic")
            transport.data_class = "TransportDataModel"

            transport.world = self.world
            transport.facility = self
            transport.data_index = self.world.register_datamodel(transport.data_class)

            self.transports.append(transport)

        # construct distribution
        self.distribution = self.world.build_logic("DistributionLogic")
        self.distribution.data_class = "DistributionDataModel"

        self.distribution.world = self.world
        self.distribution.facility = self
        self.distribution.data_index = self.world.register_datamodel(self.distribution.data_class)

    def initialize(self):
        self.storage.initialize(self.configs.get("storage", {}))
        self.distribution.initialize(self.configs.get("distribution", {}))

        transports_conf = self.configs["transports"]

        for index, transport in enumerate(self.transports):
            transport.initialize(transports_conf[index])

