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

    def _action_shaping(self, decision_event: DecisionEvent, model_action: int) -> int:
        # model action: num from vessel to port
        if model_action < 0:
            action_type = ActionType.LOAD
            quantity = min(decision_event.action_scope.load, -model_action)
        else:
            action_type = ActionType.DISCHARGE
            quantity = min(decision_event.action_scope.discharge, model_action)

        env_action = Action(
            vessel_idx=decision_event.vessel_idx,
            port_idx=decision_event.port_idx,
            quantity=int(quantity),
            action_type=action_type
        )
        return env_action

    def choose_action(
        self,
        decision_event: DecisionEvent,
        finished_events: list,
        snapshot_list: list,
        initial_port_empty: dict = None,
        initial_vessel_empty: dict = None,
        initial_vessel_full: dict = None,
    ) -> Action:

        vessel_arrival, orders, return_empty, vessel_full_delta = self._forecast_data(
            finished_events, snapshot_list, decision_event.tick
        )

        model_action = self._algorithm.choose_action(
            current_tick=decision_event.tick,
            vessel_code=self._vessel_idx2name[decision_event.vessel_idx],
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

        env_action = self._action_shaping(decision_event, model_action)

        return env_action

    def reset(self):
        """Reset agent"""
        self._algorithm.reset()
        self._forecaster.reset()
