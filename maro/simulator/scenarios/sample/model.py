from maro.simulator.scenarios.modelbase import ModelBase, IntAttribute, FloatAttribute
from maro.simulator.frame import FrameNodeType, Frame

STATIC_NODE = FrameNodeType.STATIC
DYNAMIC_NODE = FrameNodeType.DYNAMIC

## STEP 1: define your data model

# ModelBase provide a simple way to access Frame (but it not require to use it, refer to ECR scenario to see a tough way)

# NOTE: due to implementation issue, attributes registered for both static and dynamic nodes (fixed at next version)
class SampleStaticModel(ModelBase):
    # the node type for attribute used to access value
    a = IntAttribute(STATIC_NODE) # add an int attribute "a" with 1 slot (default)
    b = FloatAttribute(STATIC_NODE, slot_num=2)

    # NOTE: since your implementation of nodes is an array, so we need the index instead of id (or other identifier)
    # but you can create your own mapping out-side
    def __init__(self, frame: Frame, index: int):
        super().__init__(frame, index)

    # NOTE: value changed callbacks are binded automatically if name match "_on_<attribute name>_changed"
    def _on_a_changed(self, slot_index: int, new_val):
        print(f"value of a changed to {new_val} at slot {slot_index}")

class SampleDynamicModel(ModelBase):
    a = IntAttribute(DYNAMIC_NODE) # attribute 'a' is registered at static model, here it is a value accessor, and will be ignored when building frame
    c = IntAttribute(DYNAMIC_NODE)
    d = FloatAttribute(DYNAMIC_NODE)

    def __init__(self, frame: Frame, index: int):
        super().__init__(frame, index)
