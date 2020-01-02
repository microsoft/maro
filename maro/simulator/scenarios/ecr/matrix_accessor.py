# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.simulator.graph import Graph


class GraphMatrixAccessor:
    """
    A simple wrapper to access graph matrix attributes
    """
    def __init__(self, graph: Graph, attr_name: str):
        self._graph = graph
        self._attr_name = attr_name

    def __getitem__(self, item: slice):
        assert type(item) is slice

        row_idx = item.start
        column_idx = item.stop

        return self._graph.get_int_matrix_value(self._attr_name, row_idx, column_idx)

    def __setitem__(self, key: slice, value:int):
        assert type(key) is slice

        row_idx = key.start
        column_idx = key.stop

        return self._graph.set_int_matrix_value(self._attr_name, row_idx, column_idx, value)