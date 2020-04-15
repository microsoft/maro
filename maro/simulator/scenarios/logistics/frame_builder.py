from maro.simulator.frame import Frame, FrameAttributeType

INT = FrameAttributeType.INT
FLOAT = FrameAttributeType.FLOAT

def build(cell_num: int):
    frame = Frame(cell_num, 0)

    reg_attr = frame.register_attribute

    # markus: OBSERVATION equivalent

    reg_attr("stock", INT, 1) 

    reg_attr("demand", INT, 1) 
    
    frame.setup()

    return frame