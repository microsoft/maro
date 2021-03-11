from .base import DataModelBase

from maro.backends.frame import node, NodeBase, NodeAttribute
from maro.backends.backend import AttributeType


@node("facility")
class FacilityDataModel(DataModelBase):

    test = NodeAttribute(AttributeType.Int)

    def __init__(self):
        super(FacilityDataModel, self).__init__()

    def initialize(self, configs: dict):
        self.reset()

    def reset(self):
        super(FacilityDataModel, self).reset()
