from maro.backends.frame import FrameBase, FrameNode, NodeAttribute, NodeBase, node

TOTAL_PRODUCT_CATEGORIES = 10
TOTAL_STORES = 8
TOTAL_WAREHOUSES = 2
TOTAL_SNAPSHOT = 100


@node("warehouse")
class Warehouse(NodeBase):
    inventories = NodeAttribute("i", TOTAL_PRODUCT_CATEGORIES)
    shortages = NodeAttribute("i", TOTAL_PRODUCT_CATEGORIES)

    def __init__(self):
        self._init_inventories = [100 * (i + 1) for i in range(TOTAL_PRODUCT_CATEGORIES)]
        self._init_shortages = [0] * TOTAL_PRODUCT_CATEGORIES

    def reset(self):
        self.inventories[:] = self._init_inventories
        self.shortages[:] = self._init_shortages


@node("store")
class Store(NodeBase):
    inventories = NodeAttribute("i", TOTAL_PRODUCT_CATEGORIES)
    shortages = NodeAttribute("i", TOTAL_PRODUCT_CATEGORIES)
    sales = NodeAttribute("i", TOTAL_PRODUCT_CATEGORIES)

    def __init__(self):
        self._init_inventories = [10 * (i + 1) for i in range(TOTAL_PRODUCT_CATEGORIES)]
        self._init_shortages = [0] * TOTAL_PRODUCT_CATEGORIES
        self._init_sales = [0] * TOTAL_PRODUCT_CATEGORIES

    def reset(self):
        self.inventories[:] = self._init_inventories
        self.shortages[:] = self._init_shortages
        self.sales[:] = self._init_sales


class RetailFrame(FrameBase):
    warehouses = FrameNode(Warehouse, TOTAL_WAREHOUSES)
    stores = FrameNode(Store, TOTAL_STORES)

    def __init__(self):
        # If your actual frame number was more than the total snapshot number,
        # the old snapshots would be rolling replaced.
        super().__init__(enable_snapshot=True, total_snapshot=TOTAL_SNAPSHOT)


retail_frame = RetailFrame()

# Fulfill the initialization values to the backend memory.
for store in retail_frame.stores:
    store.reset()

# Fulfill the initialization values to the backend memory.
for warehouse in retail_frame.warehouses:
    warehouse.reset()

# Take a snapshot of the first tick frame.
retail_frame.take_snapshot(0)
snapshot_list = retail_frame.snapshots
print(f"Max snapshot list capacity: {len(snapshot_list)}")

# Query sales, inventory information of all stores at first tick, len(snapshot_list["store"]) equals to TOTAL_STORES.
all_stores_info = snapshot_list["store"][0::["sales", "inventories"]].reshape(TOTAL_STORES, -1)
print(f"All stores information at first tick (numpy array): {all_stores_info}")

# Query shortage information of first store at first tick.
first_store_shortage = snapshot_list["store"][0:0:"shortages"]
print(f"First store shortages at first tick (numpy array): {first_store_shortage}")

# Query inventory information of all warehouses at first tick,
# len(snapshot_list["warehouse"]) equals to TOTAL_WAREHOUSES.
all_warehouses_info = snapshot_list["warehouse"][0::"inventories"].reshape(TOTAL_WAREHOUSES, -1)
print(f"All warehouses information at first tick (numpy array): {all_warehouses_info}")

# Add fake shortages to first store.
retail_frame.stores[0].shortages[:] = [i + 1 for i in range(TOTAL_PRODUCT_CATEGORIES)]
retail_frame.take_snapshot(1)

# Query shortage information of first and second store at first and second tick.
store_shortage_history = snapshot_list["store"][[0, 1]: [0, 1]: "shortages"].reshape(2, -1)
print(f"First and second store shortage history at the first and second tick (numpy array): {store_shortage_history}")
