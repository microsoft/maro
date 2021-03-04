
class UnitBase:
    # class name of data model
    data_class: str = None

    # index of the data model instance in frame
    data_index: int = None

    # current world.
    world = None

    # which facility belongs to
    facility = None

    # configurations of this unit
    configs: dict = None

    # id of this unit
    id: int = None

    def __init__(self):
        # data model instance, it is None until the initialize function called.
        self._data = None

    @property
    def data(self):
        """Data model install related to this unit, available after initialized function called."""
        if self._data is None:
            self._data = self.world.get_data_instance(self.data_class, self.data_index)

        return self._data

    def initialize(self, configs: dict):
        """Initialize current unit"""
        # called after frame ready
        self.configs = configs
        self.data.initialize(configs.get("data", {}))

    def step(self, tick: int):
        # called per tick
        pass

    def get_metrics(self):
        # called per step
        pass

    def reset(self):
        # called per episode
        self.data.reset()

    def set_action(self, action):
        # called after received an action.
        pass
