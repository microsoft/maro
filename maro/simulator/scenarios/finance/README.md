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
volume of buying and selling, the investment account can be operated. You can also get the history
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

```

If necessary, attribute value can be fetched during or after episodes for subsequent analysis:

```python
stock_snapshots = env.snapshot_list['stocks']

# query account_hold_num for specified markset
sz_account_hold_num = stock_snapshots[::"account_hold_num"]

print("volume holding of sz market.")
print(sz_account_hold_num)

sz_account_net_assets_value = env.snapshot_list['account'][::"net_assets_value"]

print("net_assets_value for sz market.")
print(sz_account_net_assets_value)

```
