# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.backends.frame import FrameBase, FrameNode

from .matrix import gen_matrix
from .port import Port
from .vessel import gen_vessel_definition


def gen_cim_frame(port_num: int, vessel_num: int, stop_nums: tuple, snapshots_num: int):
    """Define and generate cim frame.

    Args:
        port_num (int): Number of ports.
        vessel_num (int): Number of vessels.
        stop_nums (tuple): Past stops number and future stop number.
    """
    vessel_cls = gen_vessel_definition(stop_nums)
    matrix_cls = gen_matrix(port_num, vessel_num)

    class CimFrame(FrameBase):
        """Our cim frame that contains vessels, ports, and a general matrix."""
        vessels = FrameNode(vessel_cls, vessel_num)
        ports = FrameNode(Port, port_num)
        matrix = FrameNode(matrix_cls, 1)

        def __init__(self):
            super().__init__(enable_snapshot=True, total_snapshot=snapshots_num)

    return CimFrame()
