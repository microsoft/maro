# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.backends.frame import NodeBase, node, NodeAttribute

@node("dummies")
class DummyNode(NodeBase):
    val = NodeAttribute("i")