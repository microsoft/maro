# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List

from maro.backends.frame import NodeAttribute, NodeBase, node


@node("clusters")
class Cluster(NodeBase):
    """Cluster node definition in frame."""
    id = NodeAttribute("i2")
    region_id = NodeAttribute("i2")
    zone_id = NodeAttribute("i2")
    data_center_id = NodeAttribute("i2")

    # Total number of machines in the cluster.
    total_machine_num = NodeAttribute("i")
    # The number of empty machines in this cluster. A empty machine means that its allocated CPU cores are 0.
    empty_machine_num = NodeAttribute("i")

    def __init__(self):
        self._id: int = 0
        self._region_id: int = 0
        self._zone_id: int = 0
        self._data_center_id: int = 0
        self._total_machine_num: int = 0

        self._cluster_type: str = ""
        self._rack_list: List[int] = []

    def set_init_state(self, id: int, region_id: int, zone_id: int, data_center_id: int, total_machine_num: int):
        """Set initialize state, that will be used after frame reset.

        Args:
            id (int): Region id.
        """
        self._id = id
        self._region_id = region_id
        self._zone_id = zone_id
        self._data_center_id = data_center_id
        self._total_machine_num = total_machine_num

        self.reset()

    def reset(self):
        """Reset to default value."""
        self.id = self._id
        self.region_id = self._region_id
        self.zone_id = self._zone_id
        self.data_center_id = self._data_center_id
        self.total_machine_num = self._total_machine_num

        self.empty_machine_num = self.total_machine_num

    @property
    def rack_list(self) -> List[int]:
        return self._rack_list

    @rack_list.setter
    def rack_list(self, rack_list: List[int]):
        self._rack_list = rack_list

    @property
    def cluster_type(self) -> str:
        return self._cluster_type

    @cluster_type.setter
    def cluster_type(self, cluster_type: str):
        self._cluster_type = cluster_type
