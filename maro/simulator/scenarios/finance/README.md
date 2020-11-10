# Finance scenario in MARO

## Introduction

The Finance scenario simulates securities trading in financial markets.
Imagine you invest in a finance market. For example, CN Shanghai, NASDAQ.
You can develop a strategy for buying and selling stocks, which try to maximize revenue.
So, your investment can keep increasing no matter how the market goes up or down.

## Environment Modeling

To simulate real markets, we implement the typical order types, cost of the trading.

A typical trading process is:

![Order_Workflow](../../../../docs/source/images/order.png "Order_Workflow")

So, you can focus on developing the strategy of trading. You can specify several
episodes of a time range. In every episode, the environment step by tick(day, minute or
event). Every tick the environment will output a group of decision events, includes the securities
can be traded, the opening price, closing price, high and low price of the securities, the max
amount of buying and selling, the investment account can be operated. You can also get the history
of securities. According to these pieces of information,
your strategy can decide how to operate the securities and accounts. After an episode finished, you
can improve the strategy, and start a new episode. You can also specify test episodes, which
evaluate the strategy.

## Topology

At present, we provide two different market, including China Shanghai, China Shengzhen.

### Topologies

- **test** has 1 stock market. China Shanghai.
Beginning trading day is 2019-01-01.
Init money in China Shanghai is 100,000 CNY.
Max avaliable volumes of each market is 20% of the total volume of the trade day.
Min buy and sell volume in China Shanghai is 100.
Slippage is 0.00246.
Close_tax in China Shanghai is 0.001.
Commission in China Shanghai is 0.0003 by money.

## Optimization Objectives

In the Finance scenario, the user can set the optimization goal as maximal **Average Return**.

**Average Return** is the simple mathematical average of a series of returns generated over a period of time.
An average return is calculated the same way a simple average is calculated for any set of numbers.

## Sample Code

Firstly, import necessary environment components for the finance scenario:

```python

import time
from collections import OrderedDict
from maro.simulator import Env, DecisionMode
from maro.simulator.frame import SnapshotList
from maro.simulator.scenarios.finance.common import Action, OrderMode

```

Then choose a topology and drive the environment using dummy actions:

```python

env = Env(scenario="finance", topology="demo", start_tick=start_tick, durations=durations, decision_mode=DecisionMode.Joint, snapshot_resolution=snapshot_resolution)
print(env.summary)
for ep in range(max_ep):
    metrics = None
    decision_evts = None
    is_done = False
    actions = None

    while not is_done:
        metrics, decision_evts, is_done = env.step(actions)

        ep_start = time.time()
        actions = []
        if not is_done:
            for decision_event in decision_evts:
                if decision_event.action_type == ActionType.order:
                    # stock trading decision
                    stock_index = decision_event.item
                    action_scope = decision_event.action_scope
                    last_frame_idx = tick_to_frame_index(start_tick, env.tick-1, snapshot_resolution)
                    min_amount = action_scope.buy_min
                    max_amount = action_scope.buy_max
                    sell_min_amount = action_scope.sell_min
                    sell_max_amount = action_scope.sell_max
                    supported_order_types = action_scope.supported_order
                    print(stock_index, min_amount, max_amount, sell_min_amount, sell_max_amount)
                    # qurey snapshot for stock information
                    cur_env_snap = env.snapshot_list['stocks']
                    holding = cur_env_snap[last_frame_idx:int(decision_event.item):"account_hold_num"][-1]
                    cost = cur_env_snap[last_frame_idx:int(decision_event.item):"average_cost"][-1]
                    opening_price = cur_env_snap[last_frame_idx:int(decision_event.item):"opening_price"][-1]
                    closing_price = cur_env_snap[last_frame_idx:int(decision_event.item):"closing_price"][-1]
                    highest_price = cur_env_snap[last_frame_idx:int(decision_event.item):"highest_price"][-1]
                    lowest_price = cur_env_snap[last_frame_idx:int(decision_event.item):"lowest_price"][-1]
                    adj_closing_price = cur_env_snap[last_frame_idx:int(decision_event.item):"adj_closing_price"][-1]
                    dividends = cur_env_snap[last_frame_idx:int(decision_event.item):"dividends"][-1]
                    splits = cur_env_snap[last_frame_idx:int(decision_event.item):"splits"][-1]
                    # qurey snapshot for account information
                    total_assets_value = env.snapshot_list['account'][last_frame_idx:0:"total_assets_value"][-1]
                    remaining_cash = env.snapshot_list['account'][last_frame_idx:0:"remaining_cash"][-1]
                    #print("env.tick: ", env.tick, " holding: ", holding, " cost: ", cost, "total_assets_value:", total_assets_value, "remaining_cash", remaining_cash)

                    if holding > 0:  # sub_engine_name -> market
                        action = MarketOrder(item=decision_event.item, amount=holding,
                                        direction=OrderDirection.sell, tick=env.tick)

                    else:

                        action = MarketOrder(item=decision_event.item, amount=1000,
                                        direction=OrderDirection.buy, tick=env.tick)

                elif decision_event.action_type == ActionType.cancel_order:
                    # cancel order decision
                    if len(decision_event.action_scope.available_orders) > 0:
                        for i in range(len(decision_event.action_scope.available_orders)):
                            if random.random() > 0.75:
                                action = CancelOrder(action_id=decision_event.action_scope.available_orders[i],
                                                tick=env.tick)
                                actions.append(action)
                    action = None
                actions.append(action)

    env.reset()
```

If necessary, attribute value can be fetched during or after episodes for subsequent analysis:

```python
stock_snapshots = env.snapshot_list['stocks']

# query account_hold_num for specified markset
sz_account_hold_num = stock_snapshots[::"account_hold_num"]

print("volume holding of sz market.")
print(sz_account_hold_num)

sz_account_total_assets_value = env.snapshot_list['account'][::"total_assets_value"]

print("total_assets_value for sz market.")
print(sz_account_total_assets_value)

```
