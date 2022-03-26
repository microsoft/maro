# SC state in MARO


## Env.summary

MARO通过summary属性对外提供节点相关信息和其他在环境初始化后不会改变的信息。
在supply chain场景中， summary中包括了以下部分

### unit mapping:

```python
env.summary["node_mapping"]["unit_mapping"]
```

unit id 及其对应的data model名字和索引.

### facilities:

```python
env.summary["node_mapping"]["facilities"]
```

每个facility的层次结构，如sku list, units

### skus:

```python
env.summary["node_mapping"]["skus"]
```

当前配置中的所有sku

### max_price:

```python
env.summary["node_mapping"]["max_price"]
```

当前配置中最大的price

### max_sources_per_facility：

```python
env.summary["node_mapping"]["max_sources_per_facility"]
```

## States

MARO中有两种对外提供动态状态(state)的方法。


### Metrics:

起初是为了对外提供reward相关的信息，但是也可以用来作为对外提供状态的接口，用来对外提供不能存放在snapshot list中的数据，比如字典，或者复杂的数据结构。

当前实现包含了以下内容：

#### products:

product unit相关信息(sale_mean, sale_std, pending_order_daily)。

#### facilities:

facility相关信息(in_transit_orders, pending_order)


### Snapshot list:

snapshot list是MARO中主要的对外提供状态的接口，它提供了所有节点的历史状态(默认为保存所有历史记录，可配置保存最新的N个来节省内存).返回的结果是numpy array,适合做batch操作。

snapshot list中的属性都是按照节点组织起来的，每个节点包括多个属性，每个属性可以有多个slot(array like), 同一种节点类型可以有多个instance.节点及其属性的定义可以在maro/simulator/scenarios/supply_chain/datamodels查看。

snapshot list的查询是通过slice接口实现的，形式如下：

```python
env.snapshot_list["node name"][tick(s):node index(s):attribute name(s)] -> np.array

```

该接口返回的是一个4维(tick, node, attribute, slot)的numpy数组(float).

其中:
1. node name是定义节点是通过node装饰器提供的名字，当前实现包括如下节点:

consumer, distribution, facility, manufacture, product, seller, storage, vehicle


2. tick(s): 可以是一个int, list或者None, 其中None表示查询当前所有历史记录的状态。

3. node index(s): 同tick，可以为int, list或者None，None表示查询当前节点类型的所有实例(instance). 使用中需要注意的是，节点(data model)的index和unit的id并不相同，unit的id是在facility和unit之间连续且唯一的，但是节点的index是每种data model类型内部的索引方式。
所以在实际使用过程中，通常需要得到每个unit和facility对应的index，这部分信息在env.summary中可以得到。

4. attribute name(s): 在对应节点上定义过的属性名，可以为一个str,或者List[str]


## 示例

### 示例1：通过env.summary构建facility&unit的层级信息

这个层级信息可以帮我们在之后的操作中快速索引。
更详细的可以参考examples/supply_chain/env_wrapper.py, _build_internal_helpers方法。

```python
# unit info
class UnitBaseInfo:
    id: int = None
    node_index: int = None
    config: dict = None
    summary: dict = None

    def __init__(self, unit_summary):
        self.id = unit_summary["id"]
        self.node_index = unit_summary["node_index"]
        self.config = unit_summary.get("config", {})
        self.summary = unit_summary

    def __getitem__(self, key, default=None):
        if key in self.summary:
            return self.summary[key]

        return default

# facility id -> {
#    data_model_index: int,
#    storage: UnitBaseInfo
#    distribution: UnitBaseInfo
#    product_id: {
#        consumer: UnitBaseInfo
#        seller: UnitBaseInfo
#        manufacture: UnitBaseInfo
#   }
#}
facility_levels = {}

# 默认env.summary包含node_mapping, node_detail 和 event_payload3个部分,
# 这里只需要node——mapping
summary = env.summary["node_mapping"]

for facility_id, facility in summary["facilities"].items():
    facility_levels[facility_id] = {
        "node_index": facility["node_index"],
        "config": facility["configs"],
        "upstreams": facility["upstreams"],
        "skus": facility["skus"]
    }

    # facility所属的unit都挂在units下面。
    units = facility["units"]

    facility_levels[facility_id]["storage"] = UnitBaseInfo(units["storage"])
    facility_levels[facility_id]["distribution"] = UnitBaseInfo(units["distribution"])

    # 所有的product unit
    product_units = units["products"]

    if product_units:
        for product_id, product in products.items():
            # product unit 本身也包含state
            product_info = {
                "product": UnitBaseInfo(product)
            }

            # 每个product unit可能包括下面3个unit
            # 注意，为了简单我们没有检查对应的key时候存在！
            product_info["seller"] = UnitBaseInfo(product["seller"])
            product_info["consumer"] = UnitBaseInfo(product["consumer"])
            product_info["manufacture"] = UnitBaseInfo(product["manufacture"])

            # 这里我们用product_id作为product 的key，可按照需求更改
            facility_levels[product_id] = product_info
```

### 示例2：通过env.summary构建unit id到node index的索引表

实际上，在示例的遍历过程中，我们就已经可以得到unit及其对应的node index索引表了，如果你不在意层级关系的话，可以通过unit_mapping快速得到这个索引。

```python

# unit_mapping是一个字典，key是unit id, value是(data model name, data model node index, facility id)类型的tuple。

summary = env.summary["node_mapping"]

unitid2index_mapping = {}

for unit_id, unit_detail in summary["unit_mapping"].items():
    unitid2index_mapping[unit_id] = unit_detail[1]

```

### 示例3：在state shaping过程中查询seller的销售和需求的历史，时间长度为hist_len

```python

# 模拟器当前时间
cur_tick = env.tick

# 需要查询的历史长度
hist_len = 4

# 历史长度对象当前时间的时间序列
ticks = [cur_tick - i for i in range(hist_len-1, -1, -1)]

# 查询seller节点的过去4（含当前）个tick的sold和demand值
# NOTE:因为这两个是都是整数，所以做一次类型转换
seller_states =env.snapshot_list["seller"][ticks::("sold", "demand")].astype(np.int)

# 结果应为4为numpy array
# 假设我们有2个seller
"""
[
    [
        [
            [0.0], # sold (slot = 1)
            [0.0]  # demand (slot = 1)
        ], # seller 0
        [...]  # seller 1
    ], # tick 0
    [
        [...],
        [...]
    ], # tick 1
    [
        [...],
        [...]
    ], # tick 2
    [
        [...],
        [...]
    ]  # tick 3 (latest)
]
"""

# 这样得到的结果就是所有的seller unit对应的销售和需求历史。

# 假设我们当前需要的seller unit的data model index 为 1 的话。
cur_seller_node_index = 1

# 那么当前seller的销售和需求历史分别为：
cur_seller_hist = seller_states[:, cur_seller_node_index, :]

# 第二个参数为0，是因为sold是我们查询的第一个属性
sale_hist = cur_seller_hist[:, 0].flatten()
demand_hist = cur_seller_hist[:, 1].flatten()

```

### 示例4：计算unit或facility的balance sheet

详细的可以参考examples/supply_chain/env_wrapper.py中的BalanceSheetCalculator类。

```python

# 假设我们需要计算seller, consumer, manufacture的balance sheet.
# 实际情况需要用这3个计算出对应的product unit的balance sheet,这里只是作为示例

# 计算所需要属性
consumer_features = ("id", "order_cost", "order_product_cost")
seller_features = ("id", "sold", "demand", "price", "backlog_ratio")
manufacture_features = ("id", "manufacture_quantity", "product_unit_cost")

# 对应的3种data model snapshot list
consumer_ss = env.snapshot_list["consumer"]
seller_ss = env.snapshot_list["seller"]
manufacture_ss = env.snapshot_list["manufacture"]

# 当前时间
tick = env.tick

# 3种unit对应的所有实例的state
# 这里用len(features)做reshape的原因是，当前用到的属性都是slots=1
# 又因为我们的tick数量为1，这样reshape之后, 每行对应一个unit的实例，每列对应一个属性
consumer_states = consumer_ss[tick::consumer_features].flatten().reshape(-1, len(consumer_features))

seller_states = seller_ss[tick::seller_features].flatten().reshape(-1, len(seller_features))

man_states = manufacture_ss[tick::manufacture_features].flatten().reshape(-1, len(manufacture_features))

# balance sheet计算，通常balance sheet 包含profit和loss两部分，这里分开保存。

# consumer部分
# loss = -1 * (order_cost + order_product_cost)
consumer_loss = -1 * (consumer_states[:, 1] + consumer_states[:, 2])

# discount在代码里似乎没有用到
reward_discount = 0

# consumer step reward
consumer_reward = consumer_loss + consumer_profit * reward_discount

# seller部分
# profit = sold * price
seller_profit = seller_states[:, 1] * seller_states[:, 3]

# loss = -1 * demand * price * backlog_ratio
seller_loss = -1 * seller_states[:, 2] * seller_states[:, 3] * seller_states[:, 4]

seller_reward = seller_profit + seller_loss

# manufacture部分
# profit = 0
# loss = manufacture_number * cost
man_loss = -1 * man_states[:, 1] * man_states[:, 2]

man_reward = man_loss

# 这样我们就用numpy的batch操作完成3种unit的balance sheet和reward计算，
# 后续需要我们按照product/facility对这些结果做聚合, 这需要类似示例1这样的层级结构，具体可参考现有代码。

```
