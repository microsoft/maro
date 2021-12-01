# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.backends.frame import FrameBase, FrameNode

from .carrier import Carrier
from .route import Route


def gen_oncall_routing_frame(route_num: int, snapshots_num: int):

    class OncallRoutingFrame(FrameBase):
        carriers = FrameNode(Carrier, route_num)
        routes = FrameNode(Route, route_num)

        def __init__(self):
            super().__init__(enable_snapshot=True, total_snapshot=snapshots_num)

    return OncallRoutingFrame()
