# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.backends.frame import NodeAttribute, NodeBase, node


@node("cluster")
class Cluster(NodeBase):
    """Cluster node definition in frame."""
    id = NodeAttribute("i2")

    def __init__(self):
        self._id: int = 0

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

    @property
    def om_list(self) -> List[int]:
        return self._pm_list
