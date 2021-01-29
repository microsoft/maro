# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List

from maro.backends.frame import NodeAttribute, NodeBase, node


@node("data_centers")
class DataCenter(NodeBase):
    """Cluster node definition in frame."""
    id = NodeAttribute("i2")
    region_id = NodeAttribute("i2")
    zone_id = NodeAttribute("i2")

    total_machine_num = NodeAttribute("i")

    def __init__(self):
        self._id: int = 0

        self._name: str = ""
        self._cluster_list: List[int] = []

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

    @property
    def cluster_list(self) -> List[int]:
        return self._cluster_list

    @cluster_list.setter
    def cluster_list(self, cluster_list: List[int]):
        self._cluster_list = cluster_list

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name
