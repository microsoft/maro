# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.s

from maro.backends.frame import FrameNode, FrameBase

from .station import Station, gen_matrices_node_definition

def build_frame(station_num: int, snapshots_num: int):
    """Function to build citi_bike Frame"""
    matrices_cls = gen_matrices_node_definition(station_num)

    class CitibikeFrame(FrameBase):
        stations = FrameNode(Station, station_num)
        matrices = FrameNode(matrices_cls, 1) # for adj frame, we only need 1 node to hold the data

        def __init__(self):
            super().__init__(enable_snapshot=True, total_snapshot=snapshots_num)

    return CitibikeFrame()