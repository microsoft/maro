import numpy as np

from typing import List, Dict
from maro.simulator import AbsEnv
from collections import namedtuple, defaultdict
from .relation import SnapshotRelationManager, SnapshotRelationTree
from .nodewrapper import SnapshotNodeWrapper
from .snapshotcache import PerTypeSnapshotCache, PerInstanceSnapshotCache

from .dynamicobject import DynamicObject


_AttrDef = namedtuple("_AttrDef", ("name", "type", "slots", "is_list", "is_const"))


class SnapshotWrapper:
    def __init__(
            self,
            env: AbsEnv,
            id_name: str = "id",
            parent_id_name: str = "parent_id",
            cache_type="instance"):

        # build the cache
        if cache_type == "instance":
            self._cache = PerInstanceSnapshotCache()
        elif cache_type == "type":
            self._cache = PerTypeSnapshotCache()

        self._env = env

        # note type class, key is node name, value is the class type
        self.node_class_dict = {}

        # node instance for all node in snapshot
        self._node_instances = defaultdict(list)

        # mapping from id to node instance
        self._id2node_dict = {}

        # name used as unique id for node instance,
        # we use this to construct the level
        self._id_name = id_name

        # name used as parent id for parent
        self._parent_id_name = parent_id_name

        # is node instance is initialized
        self._nodes_initialized = False

        # build node class type first
        self._build_node_types()

        self._relations = SnapshotRelationManager(self._env.summary["node_mapping"]["edges"])

    def step(self, action: object):
        results = self._env.step(action)

        self._init_node_instance()

        return results

    def get_node_instances(self, node_name: str) -> list:
        return self._node_instances[node_name]

    def get_relation(self, relation_name: str) -> SnapshotRelationTree:
        return self._relations[relation_name]

    def _build_node_types(self):
        """This is called after env construction to build node class types"""
        summary = self._env.summary

        nodes_attr_list = {}

        for node_name, node_def in summary['node_detail'].items():
            node_number = node_def['number']
            attributes_def = node_def['attributes']

            attribute_list = []

            for attr_name, attr_def in attributes_def.items():
                attribute_list.append(_AttrDef(
                    attr_name,
                    attr_def['type'],
                    attr_def['slots'],
                    attr_def['is_list'],
                    attr_def['is_const'],
                ))

            node_cls = type(
                node_name,
                (SnapshotNodeWrapper,),
                {
                    "maro_attributes": attribute_list
                })

            self.node_class_dict[node_name] = node_cls
            nodes_attr_list[node_name] = attribute_list

            for i in range(node_number):
                node_inst = node_cls(self._env, i, self._cache)

                self._node_instances[node_name].append(node_inst)

        self._cache.init(self._env, nodes_attr_list)

    def _init_node_instance(self):
        """This function is called after first step, to get id and parent_id.
        """
        if not self._nodes_initialized:
            self._nodes_initialized = True

            for node_name, node_cls_type in self.node_class_dict.items():
                attrs = node_cls_type.maro_attributes

                for attr in attrs:
                    if attr.name == self._id_name:
                        # get id from snapshots
                        id_list = self._env.snapshot_list[node_name][0::self._id_name].flatten().astype(np.int)

                        for inst in self._node_instances[node_name]:
                            inst.maro_uid = id_list[inst.maro_node_index]

                            self._id2node_dict[inst.maro_uid] = inst

                            setattr(inst, self._id_name, inst.maro_uid)

                    if attr.name == self._parent_id_name:
                        pid_list = self._env.snapshot_list[node_name][0::self._parent_id_name].flatten().astype(np.int)

                        for inst in self._node_instances[node_name]:
                            inst.maro_parent_uid = pid_list[inst.maro_node_index]
                            setattr(inst, self._parent_id_name, inst.maro_parent_uid)

            # after instance initialized, we need go through it again to construct the level.
            for node_uid, node_inst in self._id2node_dict.items():
                # only > 0 is a valid uid and parent_uid
                if node_inst.maro_parent_uid > 0:
                    parent_node_inst = self._id2node_dict[node_inst.maro_parent_uid]
                    node_inst.maro_parent = parent_node_inst

                    # add to children
                    parent_node_inst.children[node_inst.maro_node_name] = node_inst
