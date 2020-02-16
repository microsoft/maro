
def test_graph():
    from graph import Graph, GraphDataType, AttributeType, SnapshotList

    DYNAMIC_NODE = AttributeType.DYNAMIC_NODE
    STATIC_NODE = AttributeType.STATIC_NODE
    GENERAL = AttributeType.GENERAL

    g = Graph(10, 10)
    g.reg_attr(DYNAMIC_NODE, "B", GraphDataType.SHORT, 1)
    g.reg_attr(STATIC_NODE, "a", GraphDataType.BYTE, 1)
    g.reg_attr(STATIC_NODE, "C", GraphDataType.INT32, 1)
    g.reg_attr(STATIC_NODE, "D", GraphDataType.INT64, 1)
    g.reg_attr(GENERAL, "E",GraphDataType.INT64, 100)
    g.setup()

    ss = SnapshotList(3, g)
    ss.insert_snapshot()


    print("a", g.get_attr(STATIC_NODE, 0, "a", 0))
    g.set_attr(STATIC_NODE, 0, "a", 0, 1)
    print("a", g.get_attr(STATIC_NODE, 0, "a", 0))

    print("B", g.get_attr(DYNAMIC_NODE, 0, "B", 0))
    g.set_attr(DYNAMIC_NODE, 0, "B", 0, 1234)
    print("B", g.get_attr(DYNAMIC_NODE, 0, "B", 0))

    print("E", g.get_attr(GENERAL, 0, "E", 0))
    g.set_attr(GENERAL, 0, "E", 0, 444)
    print("E", g.get_attr(GENERAL, 0, "E", 0))

    ss.insert_snapshot()
    ss.insert_snapshot()
    ss.insert_snapshot()

    print(ss.get_node_attrs(STATIC_NODE, [0, 1], [0], ["a"], [0], 0))
    print(ss.get_node_attrs(DYNAMIC_NODE, [0, 1], [0], ["B"], [0], 0))
    print(ss.get_general_attr([0, 1], "E"))

    print(ss.static_nodes[[0, 1]:0:("a", 0)])
    print(ss.dynamic_nodes[[0, 1]:0:("B", 0)])
    print(ss.general[[0, 1]:"E"])

class Task:
    def __init__(self):
        self.priority = 0
        self.payload = {}

    def __repr__(self):
        return f"p: {self.priority}, i: {self.payload}"

    def __lt__(self, other):
        return self.priority > other.priority

def test_heapq():
    # try to use heapq to impl priority queue
    from heapq import heappush, heappop
    
    q = []

    for i, p in enumerate([1, 2, 3, 3, 4, 5]):
        t = Task()
        t.priority = p
        t.payload = i
    
        heappush(q, (t, len(q)))

    for it in q:
        print(it) 

    while len(q) > 0:
        t, _= heappop(q)

        print(p, t)    
test_heapq()