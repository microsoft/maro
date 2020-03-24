from maro.simulator.frame import Frame, FrameAttributeType

INT = FrameAttributeType.INT
FLOAT = FrameAttributeType.FLOAT

def build(cell_num: int):
    frame = Frame(cell_num, 0)

    reg_attr = frame.register_attribute

    reg_attr("bikes", INT, 1)
    reg_attr("docks", INT, 1)
    reg_attr("fulfillment", INT, 1)
    reg_attr("trip_requirement", INT, 1)
    reg_attr("shortage", INT, 1)
    reg_attr("capacity", INT, 1)

    # additional features
    # we split gendor into 3 fields
    reg_attr("unknown_gendors", INT, 1)
    reg_attr("males", INT, 1)
    reg_attr("females", INT, 1)

    # usertype
    reg_attr("subscriptor", INT, 1)
    reg_attr("customer", INT, 1)

    # TODO: these attributes should be a byte value with latest branch later
    reg_attr("weekday", INT, 1)
    reg_attr("temperature", FLOAT, 1) # avg temp
    reg_attr("weather", INT, 1) # 0: sunny, 1: rainy, 2: snowy， 3: sleet
    reg_attr("holiday", INT, 1) # 0: holiday, 1: not holiday
    reg_attr("extra_cost", INT, 1)

    frame.setup()

    return frame
