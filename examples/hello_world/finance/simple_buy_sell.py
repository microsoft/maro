import random
import time

from maro.simulator import DecisionMode, Env
from maro.simulator.scenarios.finance.common.common import (ActionType,
                                                            CancelOrder,
                                                            LimitOrder,
                                                            MarketOrder,
                                                            OrderDirection)
from maro.simulator.utils.common import tick_to_frame_index
from maro.utils.logger import CliLogger

LOGGER = CliLogger(name=__name__)

AUTO_EVENT_MODE = False
START_TICK = 10226  # 2019-01-01
DURATION = 100
MAX_EP = 2
SNAPSHOT_RESOLUTION = 1

def simple_buy_and_sell_strategy():
    env = Env(
        scenario="finance", topology="simple_buy_sell", start_tick=START_TICK, durations=DURATION,
        decision_mode=DecisionMode.Joint, snapshot_resolution=SNAPSHOT_RESOLUTION
    )

    LOGGER.info_green(f"{env.summary}")
    for ep in range(MAX_EP):
        env.reset()
        _, decision_events, is_done = env.step(None)

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
                    LOGGER.info_green(
                        f"<<<<    trade decision_event: asset {stock_index} buy scope \
                        {min_amount}~{max_amount} sell scope {sell_min_amount}~{sell_max_amount}"
                    )
                    # qurey snapshot for stock information
                    cur_env_snap = env.snapshot_list['stocks']
                    holding, average_cost = \
                        cur_env_snap[last_frame_idx:stock_index:["account_hold_num", "average_cost"]]
                    # simple strategy, buy and sell higher than average_cost
                    if holding > 0:
                        # Different order types reference to TODO:
                        action = LimitOrder(
                            item=stock_index, amount=sell_max_amount,
                            direction=OrderDirection.sell, tick=env.tick, limit=average_cost * 1.03
                        )
                    else:

                        action = MarketOrder(
                            item=stock_index, amount=max_amount,
                            direction=OrderDirection.buy, tick=env.tick
                        )
                    LOGGER.info_green(f"    >>>>trade order: {action.id},tick:{env.tick}")

                elif decision_event.action_type == ActionType.cancel_order:
                    # cancel order decision
                    LOGGER.info_green(
                        f"<<<<    Cancel order decision_event:\
                        {[x.id for x in decision_event.action_scope.available_orders]},tick:{env.tick}"
                    )
                    if len(decision_event.action_scope.available_orders) > 0:
                        for i in range(len(decision_event.action_scope.available_orders)):
                            if random.random() > 0.75:
                                action = CancelOrder(
                                    order=decision_event.action_scope.available_orders[i],
                                    tick=env.tick
                                )
                                LOGGER.info_green(f"    >>>>Cancel order:{action.order.id},tick:{env.tick}")
                                actions.append(action)
                    action = None
                actions.append(action)
            _, decision_events, is_done = env.step(actions)

        # LOGGER.info_green ep results
        ep_time = time.time() - ep_start

        stock_snapshots = env.snapshot_list['stocks']

        # query account_hold_num for specified markset
        sz_account_hold_num = stock_snapshots[::"opening_price"].reshape(-1, 5)

        LOGGER.info_green("Volume holding of sz market.")
        LOGGER.info_green(f"{sz_account_hold_num}")

        sz_account_net_assets_value = env.snapshot_list['account'][::"net_assets_value"]

        LOGGER.info_green("net_assets_value for sz market.")
        LOGGER.info_green(f"{sz_account_net_assets_value}")

        LOGGER.info_green(f"Ep {ep} takes {ep_time} seconds.")


if __name__ == "__main__":
    simple_buy_and_sell_strategy()
