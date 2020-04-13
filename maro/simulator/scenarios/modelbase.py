"""Wrappers to make frame accessing easily"""

from typing import Callable
from maro.simulator.frame import Frame, FrameAttributeType, FrameNodeType

INT = FrameAttributeType.INT
FLOAT = FrameAttributeType.FLOAT
INT_MAT = FrameAttributeType.INT_MAT

class BaseAttribute:
    """Base wrapper for frame attribute.
    For attributes that it can be read/set directly without slice interface if slot_num is 1"""

    # TODO: remove row and col parameters later after merged further changes, here is for compact issue
    def __init__(self, node_type: FrameNodeType, data_type: FrameAttributeType, slot_num: int = 1, row: int = 0, col: int = 0):
        assert slot_num > 0
        
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

    def __repr__(self):
        return f"<IntAttribute: node type: {self.node_type}, slot num: {self.slot_num}>"


class FloatAttribute(BaseAttribute):
    """Describe a float attribute in frame"""
    def __init__(self, node_type: FrameNodeType, slot_num: int = 1, ndigits: int = None):
        super().__init__(node_type, FLOAT, slot_num)
        
        assert ndigits is None or ndigits >= 0
        
        self._ndigits = ndigits

    # override following 2 methods to provide rounded float value
    def get_value(self, frame: Frame, name: str, node_index: int, slot: int):
        val = super().get_value(frame, name, node_index, slot)

        if self._ndigits is None:
            return val
        else:
            return round(val, self._ndigits)

    def set_value(self, frame: Frame, name: str, node_index: int, slot: int, value):
        tmp_val = value

        if self._ndigits is not None:
            tmp_val = round(value, self._ndigits)

        return super().set_value(frame, name, node_index, slot, tmp_val)

    def __repr__(self):
        return f"<FloatAttribute: node type: {self.node_type}, slot num: {self.slot_num}, ndigits: {self._ndigits}>"

# TODO: remove after we merged further changes
class IntMaxtrixAttribute(BaseAttribute):
    def __init__(self, node_type: FrameNodeType, row: int = 1, col: int = 1):
        super().__init__(node_type, INT_MAT, 0, row=row, col=col)

class FrameAttributeSliceAccessor:
    """Used to provide a way to access frame field with slice interface"""
    def __init__(self, attr: BaseAttribute, frame: Frame, index: int, name: str):
        self.attr = attr
        self.index = index
        self.name = name
        self._frame = frame
        self._value_changed_cb: Callable = None

    def on_value_changed(self, callback: Callable):
        """Callback when the attribute changed.
        Parameters of the callback:
        1. slot index
        2. new value"""
        self._value_changed_cb = callback

    def __getitem__(self, key):
        return self.attr.get_value(self._frame, self.name, self.index, key)

    def __setitem__(self, key, value):
        self.attr.set_value(self._frame, self.name, self.index, key, value)

        if self._value_changed_cb is not None:
            self._value_changed_cb(key, value)

    def __repr__(self):
        return f"<FrameAttributeSliceAccessor {self.name}, {self.attr.__repr__()}>"


class ModelBase:
    """Base wrapper to create a wrapper to access frame with some useful functions.

    Model not only contains accessor to Frame, also with Frame definitions, as we have specified the name and data type

    Examples:
        # define a data model
        class DataModel1(ModelBase):
            # define attributes

            # int attribute a with 1 slot (default)
            a = IntAttribute()

            # float attribute b with 2 slots, and the value will be round with 2 digits
            b = FloatAttribute(slot=2, ndigits=2)

            def __init__(self, frame: Frame, index: int, name: str):
                super().__init__(frame, index)

                self.your_name = name
            
            # data model will bind a callback function that if there is any function name match _on_<attribute name>_changed automatically
            def _on_b_changed(self, slot_index: int, new_value):
                pass # do something

        # after the definition, we can build frame now

        # a frame with 10 static nodes, and 10 dynamic nodes
        frame = build_frame(DataModel1, 10, 10)

        node1 = DataModel1(frame, 0, "my name")

        # get value of attribute a
        print(node1.a[0])

        # since attribute a only has 1 slot, we can get/set it without slice
        node1.a = 101
        
        print(node1.a)

        # for attribute b, we have to use slice interface
        node1.b[0] = 12.12
        node1.b[1] = 12.31

        print(node1.b[0], node1.b[1])
    """
    def __init__(self, frame: Frame, index: int):
        self._frame = frame
        self._index = index

        self._bind_attributes()

    @property
    def index(self)->int:
        return self._index

    def _bind_attributes(self):
        """Bind attributes with frame and id"""
        __dict__ = object.__getattribute__(self, "__dict__")    

        for name, attr in type(self).__dict__.items():
            # append an attribute access wrapper to current instance
            if isinstance(attr, BaseAttribute):
                # TODO: this will override exist attribute of sub-class instance, maybe a warning later
                
                # NOTE: here we have to use __dict__ to avoid infinite loop, as we overrided __getattribute__
                attr_acc = FrameAttributeSliceAccessor(attr, __dict__["_frame"], __dict__["_index"], name)

                __dict__[name] = attr_acc

                # bind a value changed callback if available, named as _on_<attr name>_changed
                cb_name = f"_on_{name}_changed"
                cb_func = getattr(self, cb_name, None)

                if cb_func is not None:
                    attr_acc.on_value_changed(cb_func)

    def __setattr__(self, name, value):
        """Used to avoid attribute overriding"""
        __dict__ = object.__getattribute__(self, "__dict__")    
     
        if name in __dict__:
            attr_acc = __dict__[name]

            if isinstance(attr_acc, FrameAttributeSliceAccessor):
                if attr_acc.attr.slot_num > 1:
                    raise "cannot set value for frame fields directly, please use slice interface instead"
                else:
                    # short-hand for attributes with 1 slot
                    attr_acc[0] = value
            else:
                __dict__[name] = value
        else:
            __dict__[name] = value

    def __getattribute__(self, name):
        __dict__ = object.__getattribute__(self, "__dict__")        

        if name in __dict__:
            attr_acc = __dict__[name]

            if isinstance(attr_acc, FrameAttributeSliceAccessor):
                if attr_acc.attr.slot_num == 1:
                    return attr_acc[0]
          
        return super().__getattribute__(name)

def build_frame(static_model_cls, static_node_num: int, dynamic_model_cls = None, dynamic_node_num: int = 0):
    """Build frame from definition of data model"""

    def reg_attr(frame: Frame, model_cls):
        assert model_cls is not None

        assert issubclass(model_cls, ModelBase)

        for name, attr in model_cls.__dict__.items():
            if isinstance(attr, BaseAttribute):
                frame.register_attribute(name, attr.data_type, attr.slot_num, attr.row, attr.col)      

    frame = Frame(static_node_num, dynamic_node_num)

    if static_model_cls is not None:
        assert static_node_num > 0

        reg_attr(frame, static_model_cls)

    if dynamic_model_cls is not None:
        assert dynamic_node_num > 0

        reg_attr(frame, dynamic_model_cls)

    frame.setup()

    return frame