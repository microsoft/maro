"""
This is wrapper to access data in Frame with helpers
"""

from maro.simulator.frame import Frame, FrameNodeType
from maro.simulator.scenarios.modelbase import ModelBase, IntAttribute, FloatAttribute

STATIC_NODE = FrameNodeType.STATIC

class Order(ModelBase):
    
    def __init__(self, frame: Frame, index: int):
        super().__init__(frame, index)