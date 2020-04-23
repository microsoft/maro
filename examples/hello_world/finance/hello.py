from maro.simulator import Env
from maro.simulator.frame import SnapshotList
from maro.simulator.scenarios.finance.common import Action, OrderMode


env = Env("finance", "test", max_tick=-1)

print("current stocks")
print(env.node_name_mapping.test_stocks)

reward, decision_event, is_done = env.step(None)

while not is_done:
    holding = env.snapshot_list.test_stocks.static_nodes[env.tick:0:("account_hold_num", 0)][-1]
    available = env.snapshot_list.test_stocks.static_nodes[env.tick:0:("is_valid", 0)][-1]
    print("holding:",holding,"available",available)
    if available == 1:
        if holding > 0:
            action = Action("test_stocks", 0, -holding, OrderMode.market_order)
        else:
            action = Action("test_stocks",0, 100, OrderMode.market_order)
    else:
        action = None
    reward, decision_event, is_done = env.step(action)

stock_snapshots: SnapshotList = env.snapshot_list.test_stocks

print("len of snapshot:", len(stock_snapshots))

stock_opening_price = stock_snapshots.static_nodes[:0:("opening_price", 0)]

print("opening price for all the ticks:")
print(stock_opening_price)

stock_account_hold_num = stock_snapshots.static_nodes[:0:("account_hold_num", 0)]

print("account hold num for all the ticks:")
print(stock_account_hold_num)

account_snapshots: SnapshotList = env.snapshot_list.account

account_total_money = account_snapshots.static_nodes[:0:("total_money", 0)]

print("account total money for all the ticks:")
print(account_total_money)