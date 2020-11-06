# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.backends.frame import FrameBase, FrameNode

from .dummy_node import DummyNode


def build_frame(total_snapshots:int):
    class DummyFrame(FrameBase):
        dummies = FrameNode(DummyNode, 10)

        def __init__(self):
            super().__init__(enable_snapshot=True, total_snapshot=total_snapshots)

    return DummyFrame()
