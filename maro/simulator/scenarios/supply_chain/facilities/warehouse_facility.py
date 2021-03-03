
from typing import List

from ..units import (
    StorageUnit,
    TransportUnit,
    DistributionUnit
)


class WarehouseFacility:
    world: object

    storage: StorageUnit
    distribution: DistributionUnit
    transports: List[TransportUnit]

    configs: dict

    def __init__(self):
        pass

    def step(self, tick: int):

        self.storage.step(tick)
        self.distribution.step(tick)

        for transport in self.transports:
            transport.step(tick)

    def build(self, configs: dict):
        self.configs = configs

        self.storage = StorageUnit()

        # TODO: from config later
        # Choose data model and logic we want to use

        # construct storage
        self.storage.datamodel_class = "StorageDataModel"
        self.storage.logic_class = "StorageLogic"

        self.storage.world = self.world
        self.storage.facility = self
        self.storage.datamodel_index = self.world.register_datamodel("StorageDataModel")
        self.storage.logic = self.world.build_logic("StorageLogic")

        # construct transport
        self.transports = []

        for facility_conf in configs["transports"]:
            transport = TransportUnit()

            transport.datamodel_class = "TransportDataModel"
            transport.logic_class = "TransportLogic"

            transport.world = self.world
            transport.facility = self
            transport.datamodel_index = self.world.register_datamodel("TransportDataModel")
            transport.logic = self.world.build_logic("TransportLogic")

            self.transports.append(transport)

        # construct distribution
        self.distribution = DistributionUnit()
        self.distribution.datamodel_class = "DistributionDataModel"
        self.distribution.logic_class = "DistributionLogic"

        self.distribution.world = self.world
        self.distribution.facility = self
        self.distribution.datamodel_index = self.world.register_datamodel("DistributionDataModel")
        self.distribution.logic = self.world.build_logic("DistributionLogic")

    def initialize(self):
        self.storage.initialize(self.configs.get("storage", {}))
        self.distribution.initialize(self.configs.get("distribution", {}))

        transports_conf = self.configs["transports"]

        for index, transport in enumerate(self.transports):
            transport.initialize(transports_conf[index])

