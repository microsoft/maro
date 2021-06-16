from maro.simulator import AbsEnv
from typing import List


class SnapshotAttributeWrapper:
    def __init__(
        self,
        node_name: str,
        node_index: int,
        attr_name: str,
        slots: int,
        is_list: bool,
        env: AbsEnv,
        node_inst,
        cache
    ):

        self.slots = slots
        self.node_name = node_name
        self.node_index = node_index
        self.name = attr_name
        self.is_list = is_list
        self._env = env
        self._node_inst = node_inst
        self._cache = cache

    @property
    def value(self):
        return self._cache.get_attribute(self.node_name, self.node_index, self.name)

    def range(self, ticks: List[int]):
        if type(ticks) == list or type(ticks) == tuple:
            return self._cache.get_attribute_range(self.node_name, self.node_index, self.name, ticks)

        return None
