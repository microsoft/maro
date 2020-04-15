"""
This is wrapper to access data in Frame with helpers
"""

from maro.simulator.frame import Frame, FrameNodeType
from maro.simulator.scenarios.modelbase import ModelBase, IntAttribute, FloatAttribute

STATIC_NODE = FrameNodeType.STATIC

# following class defined a wrapper to acess Frame, with registered an attribute "number" as how many order generated
class Stock(ModelBase):
    # define the attribute saved in Frame, and assume Order is static node in Frame definition
    # NOTE: currently we have to specified the node type as the attribute parameter to access correct value
    number = IntAttribute(STATIC_NODE)

    def __init__(self, frame: Frame, index: int):
        super().__init__(frame, index)