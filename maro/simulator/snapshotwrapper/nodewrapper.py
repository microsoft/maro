# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.simulator import AbsEnv
from .attributewrapper import SnapshotAttributeWrapper

from .dynamicobject import DynamicObject


class SnapshotNodeWrapper:
    """Base of node class, used to provide helper functions"""
    maro_node_name = None
    maro_node_index = 0
    maro_uid = None
    maro_parent_uid = None
    maro_parent = None

    # key is the node name of the child, value is the child instance list
    children: DynamicObject = None

    def __init__(self, env: AbsEnv, node_index, cache):
        self._env = env

        self._cache = cache
        self.maro_node_name = type(self).__name__
        self.maro_node_index = node_index
        self.children = DynamicObject()

        for attr_def in self.maro_attributes:
            self.__dict__[attr_def.name] = SnapshotAttributeWrapper(
                self.maro_node_name,
                self.maro_node_index,
                attr_def.name,
                attr_def.slots,
                attr_def.is_list,
                env,
                self,
                cache,
            )

    def __str__(self):
        return f"<{self.maro_node_name}, node_index: {self.maro_node_index}, uid: {self.maro_uid}, parent_id: {self.maro_parent_uid}>"

    def __repr__(self):
        return self.__str__()

    def __getattribute__(self, item):
        """Used to support get attribute value if slot number is 1"""
        d = object.__getattribute__(self, "__dict__")

        if item in d:
            a = d[item]

            if type(a) == SnapshotAttributeWrapper:
                values = a.value

                if not a.is_list and a.slots == 1:
                    return values[0]
                else:
                    return values

            return a

        return super(SnapshotNodeWrapper, self).__getattribute__(item)

    def __getitem__(self, item: slice):
        """Used to support """
        if type(item) != slice:
            return

        ticks = item.start
        attr_name = item.stop

        aw = self.__dict__.get(attr_name, None)

        if aw is not None and type(aw) == SnapshotAttributeWrapper:
            return aw.range(ticks)

        return None
