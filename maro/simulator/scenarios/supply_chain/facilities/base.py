
from abc import ABC, abstractmethod


class FacilityBase(ABC):
    # current world
    world = None

    # configuration of this facility
    configs: dict = None

    # id of this facility
    id: int = None

    # name of this facility
    name: str = None

    # sku information, same as original sku_in_stock
    # different facility may contains different data
    sku_information: dict = None

    # dictionary of upstreams
    # key is the source sku id
    # value is the list of facility id
    upstreams: dict = None

    @abstractmethod
    def step(self, tick: int):
        # called per tick
        pass

    def post_step(self, tick: int):
        pass

    @abstractmethod
    def build(self, configs: dict):
        # called to build components, but without data model instance.
        pass

    @abstractmethod
    def initialize(self):
        # called after data model instance is ready.
        pass

    @abstractmethod
    def reset(self):
        # called per episode.
        pass
