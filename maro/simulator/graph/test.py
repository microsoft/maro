
from graph import Graph, PartitionType, AttributeType, SnapshotList

g = Graph(10, 10)
g.reg_attr(PartitionType.DYNAMIC_NODE, "B", AttributeType.SHORT, 1)

g.reg_attr(PartitionType.STATIC_NODE, "a", AttributeType.BYTE, 1)
g.reg_attr(PartitionType.STATIC_NODE, "C", AttributeType.INT32, 1)
g.reg_attr(PartitionType.STATIC_NODE, "D", AttributeType.INT64, 1)
# g.reg_attr(PartitionType.GENERAL, "E",AttributeType.INT64, 100)
g.setup()

ss = SnapshotList(10, g)
ss.insert_snapshot(0)

print("a", g.get_attr(PartitionType.STATIC_NODE, 0, "a", 0))
g.set_attr(PartitionType.STATIC_NODE, 0, "a", 0, 1)
print("a", g.get_attr(PartitionType.STATIC_NODE, 0, "a", 0))

print("B", g.get_attr(PartitionType.DYNAMIC_NODE, 0, "B", 0))
g.set_attr(PartitionType.DYNAMIC_NODE, 0, "B", 0, 1234)
print("B", g.get_attr(PartitionType.DYNAMIC_NODE, 0, "B", 0))

ss.insert_snapshot(1)


# from graph import test_byte_cast

# print(test_byte_cast())