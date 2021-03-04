

from .base import UnitBase


class ConsumerUnit(UnitBase):
    def __init__(self):
        super(ConsumerUnit, self).__init__()

    def initialize(self, configs: dict):
        super(ConsumerUnit, self).initialize(configs)

    def step(self, tick: int):
        pass

    def reset(self):
        super(ConsumerUnit, self).reset()
