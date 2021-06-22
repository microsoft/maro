# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from collections import defaultdict, Iterable


class _RelationTreeNode:
    def __init__(self, uid: int):
        # key is the edge value, value if the child nodes, one node could have same child with different edge value
        # sku id -> [facility Node (facility id), ]
        self.children = defaultdict(list)
        self.parent = None
        self.uid = uid


class SnapshotRelationTree:
    """One relation in current definition."""
    def __init__(self):
        self.root = _RelationTreeNode(None)

        # flatten node from up to down
        self.flatten_nodes = []
        self._id2node = {}

    def add(self, edge_info: tuple):
        from_uid, to_uid, edge_value = edge_info

        stack = [self.root]

        src_node: _RelationTreeNode = None
        to_node: _RelationTreeNode = None

        while len(stack) > 0:
            n = stack.pop()

            if n.uid == from_uid:
                src_node = n
            elif n.uid == to_uid:
                to_node = n
            else:
                for child in n.children.values():
                    stack.extend(child)

            if src_node is not None and to_node is not None:
                break

        if src_node is None:
            # a new node, then add it as root' child
            src_node = _RelationTreeNode(from_uid)
            src_node.parent = self.root

            self.root.children[None].append(src_node)

            self._id2node[from_uid] = src_node

        if to_node is None:
            to_node = _RelationTreeNode(to_uid)

            self._id2node[to_uid] = to_node
        else:
            # if to node has parent and its root, then remove it from root
            if to_node.parent is not None and to_node.parent == self.root:
                self.root.children[None].remove(to_node)

        to_node.parent = src_node

        src_node.children[edge_value].append(to_node)

    def get_node(self, id: int) -> _RelationTreeNode:
        return self._id2node[id]

    def _flatten(self):
        # flatten the tree first to speedup further using, as this is a static tree.
        fifo_list = [n for n in self.root.children[None]]
        # key is the uid, value is bool
        masks = {}

        while len(fifo_list) > 0:
            # BAD performance!!
            n = fifo_list.pop(0)

            # if we have not processed this node
            if not masks.get(n.uid, False):
                masks[n.uid] = True

                self.flatten_nodes.append(n.uid)

                for c in n.children.values():
                    fifo_list.extend(c)

    def up_to_down(self) -> Iterable:
        if len(self.flatten_nodes) == 0:
            self._flatten()

        return self.flatten_nodes

    def down_to_up(self) -> Iterable:
        if len(self.flatten_nodes) == 0:
            self._flatten()

        return [x for x in reversed(self.flatten_nodes)]


class SnapshotRelationManager:
    """manage the relations defined in 'edges' field between node instances.

    Basically it is several trees inside.
    """
    def __init__(self, edges: dict):
        self._tree_dict = {}

        for edge_name, edge_list in edges.items():
            tree = SnapshotRelationTree()

            for edge in edge_list:
                tree.add(edge)

            self._tree_dict[edge_name] = tree

    def __getitem__(self, name: str) -> SnapshotRelationTree:
        return self._tree_dict[name]
