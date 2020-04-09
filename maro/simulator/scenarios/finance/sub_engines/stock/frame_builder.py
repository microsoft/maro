from maro.simulator.frame import Frame, SnapshotList, FrameAttributeType

INT_TYPE = FrameAttributeType.INT
FLOAT_TYPE = FrameAttributeType.FLOAT

def build_frame(stocks_num: int):
    frame = Frame(stocks_num, 0)

    reg_attr = frame.register_attribute

    reg_attr("opening_price", FLOAT_TYPE, 1)
    reg_attr("closing_price", FLOAT_TYPE, 1)
    reg_attr("highest_price", FLOAT_TYPE, 1)
    reg_attr("lowest_price", FLOAT_TYPE, 1)
    reg_attr("trade_amount", FLOAT_TYPE, 1)
    reg_attr("trade_volume", INT_TYPE, 1)
    reg_attr("trade_num", INT_TYPE, 1)
    reg_attr("daily_return", FLOAT_TYPE, 1)

    frame.setup()

    return frame