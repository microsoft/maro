import random
import time
from collections import OrderedDict

from maro.simulator import DecisionMode, Env
from maro.simulator.scenarios.finance.common.common import (ActionType,
                                                            CancelOrder,
                                                            DecisionEvent,
                                                            LimitOrder,
                                                            MarketOrder,
                                                            OrderDirection,
                                                            OrderMode,
                                                            StopLimitOrder,
                                                            StopOrder)
from maro.simulator.utils.common import tick_to_frame_index

AUTO_EVENT_MODE = False
START_TICK = 10226  # 2019-01-01
DURATION = 100
MAX_EP = 2
SNAPSHOT_RESOLUTION = 1

env = Env(scenario="finance", topology="simple_buy_sell", start_tick=START_TICK, durations=DURATION, decision_mode=DecisionMode.Joint, snapshot_resolution=SNAPSHOT_RESOLUTION)

print(env.summary)
for ep in range(MAX_EP):
    metrics, decision_events, is_done = env.step(None)

    while not is_done:
        ep_start = time.time()
        actions = []
        for decision_event in decision_events:
            if decision_event.action_type == ActionType.order:
                # stock trading decision
                stock_index = decision_event.item
                action_scope = decision_event.action_scope
                last_frame_idx = tick_to_frame_index(START_TICK, env.tick-1, SNAPSHOT_RESOLUTION)
                min_amount = action_scope.buy_min
                max_amount = action_scope.buy_max
                sell_min_amount = action_scope.sell_min
                sell_max_amount = action_scope.sell_max
                supported_order_types = action_scope.supported_order
                print("<<<<    trade decision_event", stock_index, min_amount, max_amount, sell_min_amount, sell_max_amount)
                # qurey snapshot for stock information
                cur_env_snap = env.snapshot_list['stocks']
                holding, average_cost = cur_env_snap[last_frame_idx:decision_event.item:["account_hold_num", "average_cost"]]
                
                # simple strategy, buy and sell higher than average_cost
                if holding > 0:
                    # Different order types reference to TODO:
                    action = LimitOrder(item=decision_event.item, amount=sell_max_amount,
                                    direction=OrderDirection.sell, tick=env.tick, limit=average_cost * 1.03)
                else:

                    action = MarketOrder(item=decision_event.item, amount=max_amount,
                                    direction=OrderDirection.buy, tick=env.tick)
                print(f"    >>>>trade order: {action.id},tick:{env.tick}")

            elif decision_event.action_type == ActionType.cancel_order:
                # cancel order decision
                print(f"<<<<    Cancel order decision_event:{[x.id for x in decision_event.action_scope.available_orders]},tick:{env.tick}")
                if len(decision_event.action_scope.available_orders) > 0:
                    for i in range(len(decision_event.action_scope.available_orders)):
                        if random.random() > 0.75:
                            action = CancelOrder(order=decision_event.action_scope.available_orders[i],
                                            tick=env.tick)
                            print(f"    >>>>Cancel order:{action.order.id},tick:{env.tick}")
                            actions.append(action)
                action = None
            actions.append(action)
        metrics, decision_events, is_done = env.step(actions)

    #env.reset()
    ep_time = time.time() - ep_start

stock_snapshots = env.snapshot_list['stocks']

# query account_hold_num for specified markset
sz_account_hold_num = stock_snapshots[::"account_hold_num"].reshape(-1, 5)

print("volume holding of sz market.")
print(sz_account_hold_num)

sz_account_total_assets_value = env.snapshot_list['account'][::"total_assets_value"]

print("total_assets_value for sz market.")
print(sz_account_total_assets_value)

print("total second:", ep_time)
