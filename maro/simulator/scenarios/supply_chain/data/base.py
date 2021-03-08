

from abc import abstractmethod
from maro.backends.backend import AttributeType
from maro.backends.frame import NodeBase, NodeAttribute


class DataModelBase(NodeBase):
    # id of related unit
    id = NodeAttribute(AttributeType.Int)

    # id of facility this unit belongs to
    facility_id = NodeAttribute(AttributeType.Int)

    def __init__(self):
        self._unit_id = 0
        self._facility_id = 0

    @abstractmethod
    def initialize(self, configs):
        """Initialize the fields with configs, the config should be a dict."""
        # called from unit after frame is ready.
        pass

    def reset(self):
        """Reset after each episode"""
        # called per episode.
        self.id = self._unit_id
        self.facility_id = self._facility_id

    def set_id(self, unit_id: int, facility_id: int):
        self._unit_id = unit_id
        self._facility_id = facility_id
