# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Tuple

from maro.backends.frame import FrameBase, FrameNode, NodeBase


def build_frame(enable_snapshot: bool, total_snapshots: int, nodes: List[Tuple[NodeBase, str, int]]):
    class Frame(FrameBase):
        def __init__(self):
            # Inject the node definition to frame to support add node dynamically.
            for node_cls, name, number in nodes:
                setattr(Frame, name, FrameNode(node_cls, number))

            super().__init__(enable_snapshot=enable_snapshot, total_snapshot=total_snapshots, backend_name="dynamic")

    return Frame()
