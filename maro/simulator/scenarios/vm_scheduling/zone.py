# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List

from maro.backends.frame import NodeAttribute, NodeBase, node


@node("zones")
class Zone(NodeBase):
    """Zone node definition in frame."""
    id = NodeAttribute("i2")
    region_id = NodeAttribute("i2")

    total_machine_num = NodeAttribute("i")
    empty_machine_num = NodeAttribute("i")

    def __init__(self):
        self._id: int = 0
        self._region_id: int = 0
        self._total_machine_num: int = 0

        self._name: str = ""
        self._data_center_list: List[int] = []

    def set_init_state(self, id: int, region_id: int, total_machine_num: int):
        """Set initialize state, that will be used after frame reset.

        Args:
            id (int): Region id.
        """
        self._id = id
        self._region_id = region_id
        self._total_machine_num = total_machine_num

        self.reset()

    def reset(self):
        """Reset to default value."""
        self.id = self._id
        self.region_id = self._region_id
        self.total_machine_num = self._total_machine_num

        self.empty_machine_num = self.total_machine_num

    @property
    def data_center_list(self) -> List[int]:
        return self._data_center_list

    @data_center_list.setter
    def data_center_list(self, data_center_list: List[int]):
        self._data_center_list = data_center_list

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name) -> str:
        self._name = name
