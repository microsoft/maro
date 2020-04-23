from maro.simulator import Env
from maro.simulator.frame import SnapshotList
from maro.simulator.scenarios.finance.common import Action, OrderMode


env = Env("finance", "test", max_tick=-1)

print("current stocks")
print(env.node_name_mapping.test_stocks)

reward, decision_event, is_done = env.step(None)

while not is_done:
    reward, decision_event, is_done = env.step(Action("test_stocks", 0, 10000, OrderMode.market_order))

stock_snapshots: SnapshotList = env.snapshot_list.test_stocks

print("len of snapshot:", len(stock_snapshots))

stock_opening_price = stock_snapshots.static_nodes[:0:"opening_price"]

print("opening price for all the ticks:")
print(stock_opening_price)

stock_account_hold_num = stock_snapshots.static_nodes[:0:"account_hold_num"]

print("account hold num for all the ticks:")
print(stock_account_hold_num)
