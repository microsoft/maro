import random
import time

from maro.simulator import DecisionMode, Env
from maro.simulator.scenarios.finance.common import (ActionType, CancelOrder,
                                                     LimitOrder, MarketOrder,
                                                     OrderDirection)
from maro.utils.logger import CliLogger

logger = CliLogger(name=__name__)

START_TICK = 10226  # 2019-01-01
DURATION = 100  # Env have 100 steps pre episode.

def simple_strategy():
    env = Env(
        scenario="finance", topology="toy.2_stocks", start_tick=START_TICK, durations=DURATION,
        decision_mode=DecisionMode.Joint
    )

    logger.info_green(f"Environment summary: {env.summary}")
    _, decision_events, is_done = env.step(None)

    while not is_done:
        ep_start = time.time()
        actions = []
        for decision_event in decision_events:
            if decision_event.action_type == ActionType.ORDER:
                # stock trading decision
                stock_index = decision_event.stock_index
                action_scope = decision_event.action_scope
                last_frame_idx = env.tick-1
                max_amount = action_scope.max_buy_volume
                sell_max_amount = action_scope.max_sell_volume
                logger.info_green(
                    f"|||| trade decision_event: {decision_event}"
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
                        direction=OrderDirection.SELL, tick=env.tick, limit=average_cost * 1.03
                    )
                else:

                    action = MarketOrder(
                        item=stock_index, amount=max_amount,
                        direction=OrderDirection.BUY, tick=env.tick
                    )
                logger.info_green(f"---- trade order: {action.id},tick:{env.tick}")

            elif decision_event.action_type == ActionType.CANCEL_ORDER:
                # cancel order decision
                logger.info_green(
                    f"|||| Cancel order decision_event: {decision_event}"
                )
                if len(decision_event.action_scope.available_orders) > 0:
                    for i in range(len(decision_event.action_scope.available_orders)):
                        if random.random() > 0.75:
                            action = CancelOrder(
                                order=decision_event.action_scope.available_orders[i],
                                tick=env.tick
                            )
                            logger.info_green(f"---- Cancel order:{action.order.id},tick:{env.tick}")
                            actions.append(action)
                action = None
            actions.append(action)
        _, decision_events, is_done = env.step(actions)

    # logger.info_green ep results
    ep_time = time.time() - ep_start

    stock_snapshots = env.snapshot_list['stocks']

    # query account_hold_num for specified markset
    sz_account_hold_num = stock_snapshots[::"opening_price"].reshape(-1, 5)

    logger.info_green("Volume holding of sz market.")
    logger.info_green(f"{sz_account_hold_num}")

    sz_account_net_assets_value = env.snapshot_list['account'][::"net_assets_value"]

    logger.info_green("Net assets value for sz market.")
    logger.info_green(f"{sz_account_net_assets_value}")

    logger.info_green(f"The topology toy.2_stocks takes {ep_time} seconds.")


if __name__ == "__main__":
    simple_strategy()
