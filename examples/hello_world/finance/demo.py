import time
import random
from collections import OrderedDict
from maro.simulator import Env, DecisionMode
from maro.simulator.scenarios.finance.common import Action, OrderMode, ActionType, DecisionEvent

auto_event_mode = False
start_tick = 10226
durations = 100
max_ep = 1

env = Env(scenario="finance", topology="demo", start_tick=start_tick, durations=durations, decision_mode=DecisionMode.Joint, snapshot_resolution=1)

print(env.summary)
for ep in range(max_ep):
    metrics = None
    decision_evt: DecisionEvent = None
    is_done = False
    actions = None

    while not is_done:
        metrics, decision_evt, is_done = env.step(actions)

        
        ep_start = time.time()
        actions = []
        if not is_done:
            for decision_event in decision_evt:
                if decision_event.action_type == ActionType.order:
                    print("decision_event",decision_event)
                    # stock trading decision
                    # qurey snapshot for information
                    cur_env_snap = env.snapshot_list.sz_stocks['stocks']
                    holding = cur_env_snap[env.tick-1:int(decision_event.item):"account_hold_num"][-1]
                    # available = cur_env_snap[env.tick-1:int(decision_event.item):"is_valid"][-1]
                    cost = cur_env_snap[env.tick-1:int(decision_event.item):"average_cost"][-1]

                    total_money = env.snapshot_list.sz_stocks['sub_account'][env.tick-1:0:"total_money"][-1]
                    print("env.tick: ", env.tick, " holding: ", holding, " cost: ", cost, "total_money:", total_money)

                    # if available == 1:
                    if holding > 0:# sub_engine_name -> market
                        action = Action(sub_engine_name=decision_event.sub_engine_name, item_index=decision_event.item, number=-holding,
                                        action_type=ActionType.order, tick=env.tick, order_mode=OrderMode.market_order)
                    else:

                        action = Action(sub_engine_name=decision_event.sub_engine_name, item_index=decision_event.item, number=1000,
                                        action_type=ActionType.order, tick=env.tick, order_mode=OrderMode.market_order)
                    # else:
                    #     action = None
                elif decision_event.action_type == ActionType.transfer:
                    action = None
                    # account transfer decision
                    # cur_env_snap = getattr(env.snapshot_list, decision_event.sub_engine_name)
                    # sub_account_money = cur_env_snap.dynamic_nodes[env.tick:0:"remaining_money"][-1]
                    # available = 1

                    # main_account_money = env.snapshot_list.account.static_nodes[env.tick:0:"remaining_money"][-1]
                    # if env.tick % 2 == 0:
                    #     action = Action(sub_engine_name=decision_event.sub_engine_name, number=-100,
                    #                         action_type=ActionType.transfer, tick=env.tick)
                    # else:
                    #     action = Action(sub_engine_name=decision_event.sub_engine_name, number=100,
                    #                         action_type=ActionType.transfer, tick=env.tick)
                elif decision_event.action_type == ActionType.cancel_order:
                    # cancel order decision
                    print(f"Cancel order decision_event:{decision_event.action_scope},tick:{env.tick}")
                    if len(decision_event.action_scope) > 0:
                        for i in range(len(decision_event.action_scope)):
                            if random.random() > 0.75:
                                action = Action(sub_engine_name=decision_event.sub_engine_name, item_index=decision_event.action_scope[i],
                                            action_type=ActionType.cancel_order, tick=env.tick)
                                actions.append(action)
                    action = None
                actions.append(action)


    #env.reset()
    ep_time = time.time() - ep_start

stock_snapshots = env.snapshot_list.sz_stocks['stocks']

print("len of snapshot:", len(stock_snapshots))

# stock_opening_price = stock_snapshots.static_nodes[:0:"opening_price"]

# print("opening price for all the ticks:")
# print(stock_opening_price)

# stock_closing_price = stock_snapshots.static_nodes[:0:"closing_price"]

# print("closeing price for all the ticks:")
# print(stock_closing_price)

# stock_account_hold_num = stock_snapshots.static_nodes[:0:"account_hold_num"]

# print("account cn_stocks hold num for all the ticks:")
# print(stock_account_hold_num)

# stock_snapshots: SnapshotList = env.snapshot_list.sh_stocks

# stock_account_hold_num = stock_snapshots.static_nodes[:0:"account_hold_num"]

# print("account sh_stocks hold num for all the ticks:")
# print(stock_account_hold_num)


# query remaining money for specified markset
sz_market_remaining_money = stock_snapshots[::"account_hold_num"]

print("remaining money for sz market.")
print(sz_market_remaining_money)

# account_remaining_money = env.snapshot_list.account.static_nodes[::"remaining_money"]

# print("remaining money for account")
# print(account_remaining_money)

sz_market_total_money = env.snapshot_list.sz_stocks['sub_account'][::"total_money"]

print("total_money for sz market.")
print(sz_market_total_money)


# NOTE: assets interface msht provide ticks
# assets_query_ticks = [0, 1, 2]
# account_hold_assets = env.snapshot_list.account.assets[assets_query_ticks: "sz_stocks"]

# print(f"assets of account at tick {assets_query_ticks}")
# print(account_hold_assets)

# for sub_engine_name, asset_number in account_hold_assets.items():
#     print(f"engine name: {sub_engine_name}")
#     print(f"000001, 000002")
#     print(asset_number.reshape(len(assets_query_ticks), -1))

# # print("trade history")

# print(env.snapshot_list.action_history)
# print(env.snapshot_list.finished_action)
print("total second:", ep_time)
