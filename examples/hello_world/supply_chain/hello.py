# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.simulator import Env

from examples.supply_chain.common.balance_calculator import BalanceSheetCalculator

env = Env(scenario="supply_chain", topology="super_vendor", start_tick=0, durations=100)

balance_calculator = BalanceSheetCalculator(env=env)

for ep in range(2):
    is_done = False
    action = None

    while not is_done:
        metrics, _, is_done = env.step(action)

    print("*******************************************************************")

    total_sold = env.snapshot_list["seller"][env.tick :: "total_sold"].reshape(-1)
    total_demand = env.snapshot_list["seller"][env.tick :: "total_demand"].reshape(-1)
    total_sold_ratio = total_sold / total_demand * 100
    print(f"Ep {ep}: Total sold ratio (%):")
    print(total_sold_ratio[::15])

    demand = env.snapshot_list["seller"][:1:"demand"].reshape(-1)
    print(f"Demand sample of product 1:\t{demand[::15]}")

    sold = env.snapshot_list["seller"][:1:"sold"].reshape(-1)
    print(f"Sold sample of product 1:\t{sold[::15]}")

    balance_and_reward = balance_calculator.calc_and_update_balance_sheet(tick=env.tick)
    balance_list = sorted([(i, b[0]) for i, b in balance_and_reward.items()])
    print(balance_list[::80])

    env.reset()
