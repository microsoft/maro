# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.backends.frame import FrameBase, FrameNode

from .ahu import AHU

def gen_hvac_frame(snapshots_num: int, ahu_num: int):
    class HVACFrame(FrameBase):
        ahus = FrameNode(AHU, ahu_num)

        def __init__(self):
            super().__init__(enable_snapshot=True, total_snapshot=snapshots_num)

    return HVACFrame()
