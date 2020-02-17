# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.simulator.graph import Graph, GraphAttributeType

AT_GENERAL = GraphAttributeType.GENERAL

class GraphMatrixAccessor:
    """
    A simple wrapper to access graph matrix attributes
    """
    def __init__(self, graph: Graph, attr_name: str, row_num: int, col_num: int):
        self._graph = graph
        self._attr_name = attr_name
        self._row_num = row_num
        self._col_num = col_num

    def __getitem__(self, item: slice):
        assert type(item) is slice

        row_idx = item.start
        column_idx = item.stop

        return self._graph.get_attribute(AT_GENERAL, 0, self._attr_name, row_idx * self._col_num + column_idx)

    def __setitem__(self, key: slice, value:int):
        assert type(key) is slice

        row_idx = key.start
        column_idx = key.stop

        return self._graph.set_attribute(AT_GENERAL, 0, self._attr_name, row_idx * self._col_num + column_idx, value)