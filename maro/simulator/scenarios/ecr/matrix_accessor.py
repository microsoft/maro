# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.simulator.frame import Frame


class FrameMatrixAccessor:
    """
    A simple wrapper to access frame matrix attributes
    """
    def __init__(self, frame: Frame, attr_name: str):
        self._frame = frame
        self._attr_name = attr_name

    def __getitem__(self, item: slice):
        assert type(item) is slice

        row_idx = item.start
        column_idx = item.stop

        return self._frame.get_int_matrix_value(self._attr_name, row_idx, column_idx)

    def __setitem__(self, key: slice, value:int):
        assert type(key) is slice

        row_idx = key.start
        column_idx = key.stop

        return self._frame.set_int_matrix_value(self._attr_name, row_idx, column_idx, value)