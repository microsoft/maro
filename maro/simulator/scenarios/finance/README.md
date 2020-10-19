# Finance scenario in MARO

## Introduction

The Finance scenario simulates securities trading in financial markets.
Imagine you invest in multiple finance markets. For example, US NASDAQ and CN Shanghai:
You can create a strategy which decides how to buy and sell stocks in these twoindividual markets, and when NASDAQ about to go down, transfer your investments to Shanghai.
So, your investment can keep increasing no matter how the market goes up or down.

## Environment Modeling

To simulate real markets, we implement the typical order types, cost of the trading, leverage
rate of account, and dynamic exchange rates between different currencies.

A typical trading process is:

![Order_Workflow](../../../../docs/source/images/order.png "Order_Workflow")

A typical transfer process is:

![Transfer_Workflow](../../../../docs/source/images/transfer.png "Transfer_Workflow")

So, you can focus on importing the strategy of trading and transferring. You can specify several
episodes of a time range. In every episode, the environment step by tick(day, minute or
event). Every tick the environment will output a group of decision events, includes the securities
can be traded, the opening price, closing price, high and low price of the securities, the max
amount of buying and selling, the investment account can be operated, the max amount of transferring
in and out. You can also get the history of securities. According to these pieces of information,
your strategy can decide how to operate the securities and accounts. After an episode finished, you
can improve the strategy, and start a new episode. You can also specify test episodes, which
evaluate the strategy.

## Topology

At present, we provide three different market, including China Shanghai, China Shengzhen, US NASDAQ. A simple topology use China Shanghai and US NASDAQ.

### Topologies

- **test** has 2 stock market. China Shanghai and US NASDAQ.
Beginning trading day is 2019-01-01.
Init money in main account is 1000000 CNY, in China Shanghai is 100000 CNY, in US NASDAQ is 100000 USD.
Max avaliable volumes of each market is 60% of the total volume of the trade day.
Leverage rate in China Shanghai is 2.0, whereas in US NASDAQ is 1.0.
Min leverage rate in China Shanghai is 1.2, whereas in US NASDAQ is 0.
Min buy and sell volume in China Shanghai is 100, whereas in US NASDAQ is 1.
Slippage is 0.00246.
Close_tax in China Shanghai is 0.001, whereas in US NASDAQ is 0.
Commission in China Shanghai is 0.0003 by money, whereas in US NASDAQ is 29.95 by purchase.

## Optimization Objectives

In the Finance scenario, the user can set the optimization goal as maximal **Average Return**.

**Average Return** is the simple mathematical average of a series of returns generated over a period of time. An average return is calculated the same way a simple average is calculated for any set of numbers.

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

env = Env(scenario='ecr', topology='5p_ssddd_l0.0', max_tick=10)

for ep in range(2):
    reward, decision_events, is_done = env.step(None)

    while not is_done:
        actions = []
        available_ticks = OrderedDict()

        for decision_event in decision_events:
            if decision_event.item >= 0:
                # stock decision
                cur_env_snap = getattr(env.snapshot_list, decision_event.sub_engine_name)
                holding = cur_env_snap.static_nodes[env.tick:decision_event.item:"account_hold_num"][-1]
                available = cur_env_snap.static_nodes[env.tick:decision_event.item:"is_valid"][-1]

                total_money = cur_env_snap.dynamic_nodes[env.tick:0:"total_money"][-1]
                print("env.tick: ", env.tick, " holding: ", holding, " available: ", available, "total_money:", total_money)

                if available == 1:
                    if holding > 0:
                        action = Action(decision_event.sub_engine_name, decision_event.item, -holding, OrderMode.market_order)
                    else:
                        action = Action(decision_event.sub_engine_name, decision_event.item, 10000000, OrderMode.market_order)
                    if decision_event.sub_engine_name not in available_ticks:
                        available_ticks[decision_event.sub_engine_name] = OrderedDict()
                    if decision_event.item not in available_ticks[decision_event.sub_engine_name]:
                        available_ticks[decision_event.sub_engine_name][decision_event.item] = []
                    available_ticks[decision_event.sub_engine_name][decision_event.item].append(env.tick)
                else:
                    action = None
            else:
                # account decision
                cur_env_snap = getattr(env.snapshot_list, decision_event.sub_engine_name)
                holding = cur_env_snap.dynamic_nodes[env.tick:0:"remaining_money"][-1]
                available = 1

                total_money = env.snapshot_list.account.static_nodes[env.tick:0:"remaining_money"][-1]
                print("env.tick: ", env.tick, " holding: ", holding, " available: ", available, "total_money:", total_money)
                if env.tick % 2 == 0:
                    action = Action(decision_event.sub_engine_name, decision_event.item, -100, OrderMode.market_order)
                else:
                    action = Action(decision_event.sub_engine_name, decision_event.item, 100, OrderMode.market_order)
            actions.append(action)
        reward, decision_events, is_done = env.step(actions)

    env.reset()
```

If necessary, attribute value can be fetched during or after episodes for subsequent analysis:

```python
stock_snapshots: SnapshotList = env.snapshot_list.test_stocks

stock_account_hold_num = stock_snapshots.static_nodes[:0:"account_hold_num"]

print("account test_stocks hold num for all the ticks:")
print(stock_account_hold_num)

stock_snapshots: SnapshotList = env.snapshot_list.us_stocks

stock_account_hold_num = stock_snapshots.static_nodes[:0:"account_hold_num"]

print("account us_stocks hold num for all the ticks:")
print(stock_account_hold_num)

zh_market_remaining_money = stock_snapshots.dynamic_nodes[:0:"remaining_money"]

print("remaining money for zh market.")
print(zh_market_remaining_money)

account_remaining_money = env.snapshot_list.account.static_nodes[::"remaining_money"]

print("remaining money for account")
print(account_remaining_money)

```
