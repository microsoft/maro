

class UnitBase:

    logic: object
    logic_class: str
    datamodel_class: str
    datamodel_index: int

    world: object

    facility: object

    def __init__(self):
        self._datamodel = None

    @property
    def datamodel(self):
        if self._datamodel is None:
            self._datamodel = self.world.get_datamodel(self.datamodel_class, self.datamodel_index)

        return self._datamodel

    def initialize(self, configs: dict):
        self.datamodel.initialize(configs.get("data", {}))

        self.logic.data = self.datamodel
        self.logic.entity = self
        self.logic.world = self.world
        self.logic.facility = self.facility

        self.logic.initialize(configs.get("logic", {}))

    def step(self, tick: int):
        if self.logic is not None:
            self.logic.step(tick)

    def reset(self):
        pass

    def set_action(self, action):
        pass
