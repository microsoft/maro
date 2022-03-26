# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.backends.backend import AttributeType
from maro.backends.frame import NodeAttribute, NodeBase


class DataModelBase(NodeBase):
    """Base of all data model of this scenario."""
    # Id of related unit or facility, 0 is invalid by default.
    id = NodeAttribute(AttributeType.Int)

    # Id of facility this unit belongs to.
    facility_id = NodeAttribute(AttributeType.Int)

    def __init__(self) -> None:
        self._unit_id = 0
        self._facility_id = 0

    def initialize(self, **kwargs) -> None:
        """Initialize the fields with configs, the config should be a dict.

        Args:
            kwargs (dict): Configuration of related data model, used to hold value to
                reset after frame reset.
        """
        # Called from unit after frame is ready.
        pass

    def reset(self) -> None:
        """Reset after each episode."""
        # Called per episode.
        self.id = self._unit_id
        self.facility_id = self._facility_id

    def set_id(self, unit_id: int, facility_id: int) -> None:
        """Used to assign id(s), so that it will be assigned after frame rest.

        Args:
            unit_id (int): Id of related unit.
            facility_id (int)： Id of this unit belongs to.
        """
        self._unit_id = unit_id
        self._facility_id = facility_id
