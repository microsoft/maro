# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List

from maro.backends.frame import NodeAttribute, NodeBase, node


@node("racks")
class Rack(NodeBase):
    """Rack node definition in frame."""
    id = NodeAttribute("i")

    region_id = NodeAttribute("i2")
    zone_id = NodeAttribute("i2")
    data_center_id = NodeAttribute("i2")
    cluster_id = NodeAttribute("i2")

    total_machine_num = NodeAttribute("i")
    empty_machine_num = NodeAttribute("i")

    def __init__(self):
        self._id: int = 0

        self._name: str = ""
        self._pm_list: List[int] = []

    def set_init_state(self, id: int):
        """Set initialize state, that will be used after frame reset.

        Args:
            id (int): Region id.
        """
        self._id = id

        self.reset()

    def reset(self):
        """Reset to default value."""
        self.id = self._id
        self._pm_list.clear()

        self._name = ""
        self.empty_machine_num = self.total_machine_num

    @property
    def pm_list(self) -> List[int]:
        return self._pm_list

    @pm_list.setter
    def pm_list(self, pm_list: List[int]):
        self._pm_list = pm_list

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name
