# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.backends.frame import NodeBase


class MatrixAttributeAccessor:
    """Wrapper for each attribute with matrix like interface.

    Args:
        node(NodeBase): Node instance the attribute belongs to.
        attribute(str): Attribute name to wrap.
        row_num(int): Result matrix row number.
        col_num(int): Result matrix column number.
    """

    def __init__(self, node: NodeBase, attribute: str, row_num: int, col_num: int):
        self._node = node
        self._attr = None
        self._attr_name = attribute
        self._row_num = row_num
        self._col_num = col_num

    @property
    def columns(self) -> int:
        """int: Column number."""
        return self._col_num

    @property
    def rows(self) -> int:
        """int: Row number."""
        return self._row_num

    def _ensure_attr(self):
        """Ensure that the attribute instance correct"""
        if self._attr is None:
            self._attr = getattr(self._node, self._attr_name, None)

        assert self._attr is not None

    def __getitem__(self, item: tuple):
        key_type = type(item)

        self._ensure_attr()

        if key_type == tuple:
            row_idx = item[0]
            column_idx = item[1]

            return self._attr[self._col_num * row_idx + column_idx]
        elif key_type == slice:
            return self._attr[:]

    def __setitem__(self, key: tuple, value: int):
        key_type = type(key)

        self._ensure_attr()

        if key_type == tuple:
            row_idx = key[0]
            column_idx = key[1]

            self._attr[self._col_num * row_idx + column_idx] = value
        elif key_type == slice:
            # slice will ignore all parameters, and set values for all slots
            self._attr[:] = value

    def get_row(self, row_idx: int) -> list:
        """Get values of a row.

        Args:
            row_idx (int): Index of target row.

        Returns:
            list: List of value for that row.
        """
        self._ensure_attr()

        start = self._col_num * row_idx
        return self._attr[start: start + self._col_num]

    def get_column(self, column_idx: int):
        """Get values of a column.

        Args:
            column_idx (int): Index of target column.

        Returns:
            list: List of value for that column.
        """
        self._ensure_attr()

        row_indices = [r * self._col_num + column_idx for r in range(self._row_num)]

        return self._attr[row_indices]
