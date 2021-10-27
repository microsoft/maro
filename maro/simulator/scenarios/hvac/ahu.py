# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.backends.frame import NodeAttribute, NodeBase, node


@node("ahus")
class AHU(NodeBase):
    # Mixed Air Temperature
    mat = NodeAttribute("f")

    # Dust Air Temperature
    dat = NodeAttribute("f")

    # Air tonnage
    at = NodeAttribute("f")

    # kW
    kw = NodeAttribute("f")

####################################

    # Dust Static Pressure Setpoint
    sps = NodeAttribute("f")

    # Discharge Air Temperature Setpoint
    das = NodeAttribute("f")


    def __init__(self):
        self._name = None

        self._mat = 0
        self._dat = 0
        self._at = 0
        self._kw = 0

        self._sps = 0
        self._das = 0

    @property
    def name(self) ->str:
        return self._name

    @property
    def idx(self) -> int:
        return self.index

    def set_init_state(
        self, name: str, mat: float, dat: float, at: float, kw: float,
        sps: float, das: float
    ):
        self._name = name
        self._mat = mat
        self._dat = dat
        self._at = at
        self._kw = kw
        self._sps = sps
        self._das = das

        self.reset()

    def reset(self):
        self.mat = self._mat
        self.dat = self._dat
        self.at = self._at
        self.kw = self._kw
        self.sps = self._sps
        self.das = self._das
