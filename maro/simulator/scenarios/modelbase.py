"""Wrappers to make frame accessing easily"""

from maro.simulator.frame import Frame, FrameAttributeType

INT = FrameAttributeType.INT
FLOAT = FrameAttributeType.FLOAT
INT_MAT = FrameAttributeType.INT_MAT

class BaseAttribute:
    """Base wrapper for frame attribute"""

    # TODO: remove row and col parameters later after merged further changes, here is for compact issue
    def __init__(self, node_type: FrameNodeType, data_type: FrameAttributeType, slot_num: int, row: int = 0, col: int = 0):
        self.node_type = node_type
        self.data_type: int = data_type
        self.slot_num: int = slot_num
        self.row: int = row
        self.col: int = col

    def set_value(self, frame: Frame, name: str, node_index: int, slot: int, value):
        frame.set_attribute(self.node_type, node_index, name, slot, value)

    def get_value(self, frame: Frame, name: str, node_index: int, slot: int):
        return frame.get_attribute(self.node_type, node_index, name, slot)

class IntAttribute(BaseAttribute):
    """Describe an int attribute in frame"""

    # TODO: though we do not need node_type for now, it is used to merge further changes that split definition between static and dynamic nodes
    def __init__(self, node_type: FrameNodeType, slot_num: int = 1):
        super().__init__(node_type, INT, slot_num)


class FloatAttribute(BaseAttribute):
    """Describe a float attribute in frame"""
    def __init__(self, node_type: FrameNodeType, slot_num: int = 1):
        super().__init__(node_type, FLOAT, slot_num)


# TODO: remove after we merged further changes
class IntMaxtrixAttribute(BaseAttribute):
    def __init__(self, node_type: FrameNodeType, row: int = 1, col: int = 1):
        super().__init__(node_type, INT_MAT, 0, row, col)

class FrameAttributeSlideAccessor:
    """Used to provide a way to access frame field with slide interface"""
    def __init__(self, attr: BaseAttribute, frame: Frame, index: int, name: str):
        self._attr = attr
        self._frame = frame
        self._index = index
        self._name = name

    def __getitem__(self, key):
        return self._attr.get_value(self._frame, self._name, self._index, key)

    def __setitem__(self, key, value):
        self._attr.set_value(self._frame, self._name, self._index, key, value)


class ModelBase:
    """Base wrapper to create a wrapper to access frame with some useful functions"""
    def __init__(self, frame: Frame, index: int):
        self._frame = frame
        self._index = index

        self._bind_fields()

    def _bind_fields(self):
        """Bind field with frame and id"""
        for name, attr in type(self).__dict__.items():
            # append an attribute access wrapper to current instance
            if isinstance(attr, BaseAttribute):
                # TODO: this will override exist attribute of sub-class instance, maybe a warning later
                self.__dict__[name] = FrameAttributeSlideAccessor(attr, self._frame, self._index, name)

    def __setattr__(self, name, value):
        """Used to avoid attribute overriding"""
        if name in self.__dict__:
            attr = object.__getattribute__(self, name)

            if attr and isinstance(attr, BaseAttribute):
                raise "cannot set value for frame fields directly, please use slice interface instead"
                
                return
        
        self.__dict__[name] = value


def build_frame(model_cls, static_node_num: int, dynamic_node_num: int):
    """Build frame from definition of data model"""
    assert model_cls is not None

    assert issubclass(model_cls, ModelBase)

    assert static_node_num >= 0
    assert dynamic_node_num >= 0

    attributes = []

    frame = Frame(static_node_num, dynamic_node_num)

    for name, attr in model_cls.__dict__.items():
        if isinstance(attr, BaseAttribute):
            frame.register_attribute(name, attr.data_type, attr.slot_num, attr.row, attr.col)

    frame.setup()

    return frame