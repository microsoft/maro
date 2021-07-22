from maro.simulator.scenarios.cim.common import Action, ActionType, DecisionEvent, ActionScope

from forecaster import Forecaster
from online_lp import OnlineLP


class LPAgent:
    """Linear Programming Agent"""

    def __init__(
        self,
        algorithm: OnlineLP,
        forecaster: Forecaster,
        vessel_idx2name: dict,
        window_size: int
    ):
        """Init Agent"""
        self._algorithm = algorithm
        self._forecaster = forecaster
        self._vessel_idx2name = vessel_idx2name
        self._window_size = window_size

    def _forecast_data(self, finished_events, snapshot_list, current_tick):
        vessel_arrival = self._forecaster.calculate_vessel_arrival(snapshot_list, current_tick, self._window_size)

        """Forecast demand/supply"""
        self._forecaster.augment_history_record(current_tick, finished_events, snapshot_list)

        orders = self._forecaster.forecast_orders(self._window_size)

        return_empty = self._forecaster.forecast_return_empty(self._window_size)

        vessel_full_delta = self._forecaster.forecast_vessel_full_delta(self._window_size)

        return vessel_arrival, orders, return_empty, vessel_full_delta

    def _action_shaping(self, scope: ActionScope, model_action: int) -> int:
        # model action: num from vessel to port
        execute_action = model_action

        execute_action = min(execute_action, scope.discharge)
        execute_action = max(execute_action, -scope.load)
        execute_action = int(execute_action)

        # execute_action: num from vessel to port
        return execute_action

    def choose_action(
        self,
        decision_event: DecisionEvent,
        finished_events: list,
        snapshot_list: list,
        initial_port_empty: dict = None,
        initial_vessel_empty: dict = None,
        initial_vessel_full: dict = None,
    ) -> Action:
        """Choose the loading/unloading action to
        perform at each port.
        """
        tick = decision_event.tick
        port_idx = decision_event.port_idx
        vessel_idx = decision_event.vessel_idx
        vessel_name = self._vessel_idx2name[decision_event.vessel_idx]

        vessel_arrival, orders, return_empty, vessel_full_delta = self._forecast_data(
            finished_events,
            snapshot_list,
            tick
        )

        model_action = self._algorithm.choose_action(
            current_tick=tick,
            vessel_code=vessel_name,
            finished_events=finished_events,
            snapshot_list=snapshot_list,
            initial_port_empty=initial_port_empty,
            initial_vessel_empty=initial_vessel_empty,
            initial_vessel_full=initial_vessel_full,
            vessel_arrival_prediction=vessel_arrival,
            order_prediction=orders,
            return_empty_prediction=return_empty,
            vessel_full_delta_prediction=vessel_full_delta,
        )

        actual_action = self._action_shaping(
            scope=decision_event.action_scope,
            model_action=model_action
        )
        action_type = (
            ActionType.LOAD if actual_action < 0 else ActionType.DISCHARGE
        )

        env_action = Action(vessel_idx, port_idx, actual_action, action_type)

        return env_action

    def reset(self):
        """Reset agent"""
        self._algorithm.reset()
        self._forecaster.reset()
