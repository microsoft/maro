# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum, IntEnum

import dgl
import torch

from maro.common import BaseAction, BaseDecisionEvent


class VertexState(IntEnum):
    Excluded = 0
    Included = 1
    Deferred = 2


class MISEnvMetrics(Enum):
    IncludedNodeCount = "Included Node Count"
    HammingDistanceAmongSamples = "Hamming Distance Among Two Samples"
    IsDoneMasks = "Is Done Masks"


class Action(BaseAction):
    def __init__(self, vertex_states: torch.Tensor) -> None:
        self.vertex_states = vertex_states


class MISDecisionPayload(BaseDecisionEvent):
    def __init__(self, graph: dgl.DGLGraph, vertex_states: torch.Tensor) -> None:
        self.graph = graph
        self.vertex_states = vertex_states
