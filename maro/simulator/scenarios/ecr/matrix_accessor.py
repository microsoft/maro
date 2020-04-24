# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.simulator.frame import Frame, FrameNodeType

AT_GENERAL = FrameNodeType.GENERAL

class FrameMatrixAccessor:
    """
    A simple wrapper to access frame matrix (general) attributes
    """
    def __init__(self, frame: Frame, attr_name: str, rows, cols):
        self._frame = frame
        self._attr_name = attr_name
        self._rows = rows
        self._cols = cols

    def __getitem__(self, item: slice):
        assert type(item) is slice

        row_idx = item.start
        column_idx = item.stop
        return self._frame[AT_GENERAL, 0, self._attr_name, self._cols * row_idx + column_idx]

    def __setitem__(self, key: slice, value:int):
        assert type(key) is slice

        row_idx = key.start
        column_idx = key.stop

        self._frame[AT_GENERAL, 0, self._attr_name, self._cols * row_idx + column_idx] = value