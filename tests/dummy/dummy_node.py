# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.backends.frame import NodeAttribute, NodeBase, node


@node("dummies")
class DummyNode(NodeBase):
    val = NodeAttribute("i")
