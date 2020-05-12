import time
from maro.simulator import Env, DecisionMode
from maro.simulator.frame import SnapshotList
from maro.simulator.scenarios.finance.common import Action, OrderMode

MAX_EP = 2

env = Env("finance", "test", max_tick=-1, decision_mode=DecisionMode.Joint)

# print("current stocks")
# print(env.node_name_mapping.test_stocks)

for ep in range(MAX_EP):
    env.reset()
    ep_start = time.time()
    reward, decision_events, is_done = env.step(None)

    while not is_done:
        actions = []

        for decision_event in decision_events:
            if decision_event.sub_engine_name == "test_stocks":
                holding = env.snapshot_list.test_stocks.static_nodes[env.tick:decision_event.item:"account_hold_num"][-1]
                available = env.snapshot_list.test_stocks.static_nodes[env.tick:decision_event.item:"is_valid"][-1]
                total_money = env.snapshot_list.account.static_nodes[env.tick-1:0:"total_money"][-1]
                #print("env.tick: ",env.tick," holding: ",holding," available: ",available, "total_money:", total_money)

                if available == 1:
                    if holding > 0:
                        action = Action("test_stocks", decision_event.item, -holding, OrderMode.market_order)
                    else:
                        action = Action("test_stocks", decision_event.item, 500, OrderMode.market_order)
                else:
                    action = None
                actions.append(action)
            elif decision_event.sub_engine_name == "us_stocks":
                holding = env.snapshot_list.us_stocks.static_nodes[env.tick:decision_event.item:"account_hold_num"][-1]
                available = env.snapshot_list.us_stocks.static_nodes[env.tick:decision_event.item:"is_valid"][-1]
                total_money = env.snapshot_list.account.static_nodes[env.tick-1:0:"total_money"][-1]
                #print("env.tick: ",env.tick," holding: ",holding," available: ",available, "total_money:", total_money)

                if available == 1:
                    if holding > 0:
                        action = Action("us_stocks", decision_event.item, -holding, OrderMode.market_order)
                    else:
                        action = Action("us_stocks", decision_event.item, 500, OrderMode.market_order)
                else:
                    action = None
                actions.append(action)
        reward, decision_events, is_done = env.step(actions)

    ep_time = time.time() - ep_start

stock_snapshots: SnapshotList = env.snapshot_list.test_stocks

# print("len of snapshot:", len(stock_snapshots))

# stock_opening_price = stock_snapshots.static_nodes[:0:"opening_price"]

# print("opening price for all the ticks:")
# print(stock_opening_price)

# stock_closing_price = stock_snapshots.static_nodes[:0:"closing_price"]

# print("closeing price for all the ticks:")
# print(stock_closing_price)

stock_account_hold_num = stock_snapshots.static_nodes[:0:"account_hold_num"]

print("account test_stocks hold num for all the ticks:")
print(stock_account_hold_num)

stock_snapshots: SnapshotList = env.snapshot_list.us_stocks

stock_account_hold_num = stock_snapshots.static_nodes[:0:"account_hold_num"]

print("account us_stocks hold num for all the ticks:")
print(stock_account_hold_num)

account_snapshots: SnapshotList = env.snapshot_list.account

account_total_money = account_snapshots.static_nodes[:0:"total_money"]

print("account total money for all the ticks:")
print(account_total_money)

# NOTE: assets interface must provide ticks
# assets_query_ticks = [0, 1, 2]
# account_hold_assets = env.snapshot_list.account.assets[assets_query_ticks: "test_stocks"]

# print(f"assets of account at tick {assets_query_ticks}")
# print(account_hold_assets)

# for sub_engine_name, asset_number in account_hold_assets.items():
#     print(f"engine name: {sub_engine_name}")
#     print(f"000001, 000002")
#     print(asset_number.reshape(len(assets_query_ticks), -1))

# print("trade history")

# print(env.snapshot_list.trade_history)

print("total second:", ep_time)
