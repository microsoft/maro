import random
import time

from maro.simulator import DecisionMode, Env
from maro.simulator.scenarios.finance.common import (
    Cancel, CancelDecisionEvent, LimitOrder, MarketOrder, OrderDecisionEvent, OrderDirection, get_cn_stock_data_tick
)
from maro.utils.logger import CliLogger

logger = CliLogger(name=__name__)

START_TICK = get_cn_stock_data_tick("2015-01-01")
DURATION = 100


def simple_strategy():
    env = Env(
        scenario="finance", topology="toy.2_stocks", start_tick=START_TICK, durations=DURATION,
        decision_mode=DecisionMode.Joint
    )

    for i in [0, 1]:
        env.reset()
        logger.info_green(f"Environment summary: {env.summary}")
        metrics, decision_events, is_done = env.step(None)

        while not is_done:
            ep_start = time.time()
            actions = []
            for decision_event in decision_events:
                if isinstance(decision_event, OrderDecisionEvent):
                    # Stock trading decision.
                    stock_index = decision_event.stock_index
                    action_scope = decision_event.action_scope
                    last_frame_idx = env.tick - 1 - START_TICK
                    buy_max_volume = action_scope.max_buy_volume
                    sell_max_volume = action_scope.max_sell_volume
                    logger.info_green(
                        f"|||| Order decision_event: {decision_event}"
                    )
                    # Qurey snapshot for stock information.
                    cur_env_snap = env.snapshot_list['stocks']
                    holding = \
                        cur_env_snap[last_frame_idx:stock_index:"account_hold_num"]
                    average_cost = \
                        cur_env_snap[last_frame_idx:stock_index:"average_cost"]
                    # Simple strategy, buy and sell higher than average_cost.
                    if holding > 0:
                        # Different order types reference to
                        # https://www.investopedia.com/ask/answers/100314/\
                        # whats-difference-between-market-order-and-limit-order.asp .
                        action = LimitOrder(
                            stock_index=stock_index, order_volume=sell_max_volume,
                            order_direction=OrderDirection.SELL, tick=env.tick, limit=average_cost * 1.03
                        )
                    else:

                        action = MarketOrder(
                            stock_index=stock_index, order_volume=int(buy_max_volume * 0.5),
                            order_direction=OrderDirection.BUY, tick=env.tick
                        )
                    logger.info_green(f"---- Order: {action}")
                    actions.append(action)

                elif isinstance(decision_event, CancelDecisionEvent):
                    # Cancel action decision.
                    logger.info_green(
                        f"|||| Cancel decision_event: {decision_event}"
                    )
                    if len(decision_event.action_scope.available_orders) > 0:
                        for i in range(len(decision_event.action_scope.available_orders)):
                            if random.random() > 0.75:
                                action = Cancel(
                                    action=decision_event.action_scope.available_orders[i],
                                    tick=env.tick
                                )
                                logger.info_green(f"---- Cancel:{action}")
                                actions.append(action)
            if len(actions) == 0:
                actions.append(None)
            metrics, decision_events, is_done = env.step(actions)
            logger.info_green(f"Current metrics:{metrics}.")

        ep_time = time.time() - ep_start

        stock_snapshots = env.snapshot_list['stocks']

        # Query average_cost for specified market.
        sz_average_cost = stock_snapshots[::"average_cost"].reshape(-1, 2)

        logger.info_green("Average_cost of sz market.")
        logger.info_green(f"{sz_average_cost}")

        # Query account_hold_num for specified market.
        sz_account_hold_num = stock_snapshots[::"account_hold_num"].reshape(-1, 2)

        logger.info_green("Account_hold_num of sz market.")
        logger.info_green(f"{sz_account_hold_num}")

        sz_account_assets_value = env.snapshot_list['account'][::"assets_value"]

        logger.info_green("Assets value for sz market.")
        logger.info_green(f"{sz_account_assets_value}")

        logger.info_green(f"The topology toy.2_stocks takes {ep_time} seconds.")


if __name__ == "__main__":
    simple_strategy()
