import numpy as np

from maro.simulator.scenarios.citi_bike.common import Action, DecisionType


class ActionShaping:

    action_scaler = 0.1

    def __init__(self):
        pass

    def __call__(self, decision_evt, choices, nums):
        if len(choices) == 1:
            choices = choices[0]
            nums = nums[0]
        if isinstance(choices, (int, np.int, np.int64)):
            station_idx = decision_evt.station_idx
            keys = list(decision_evt.action_scope.keys())
            action = self._create_action(decision_evt.type, station_idx, keys[choices], nums)
            return action
        else:
            actions = []
            for i, de in enumerate(decision_evt):
                action = self._create_action(de['station_idx'], choices[i], nums[i])
                actions.append(action)
            return actions

    def _create_action(self, decision_type, act_idx, target_idx, action):
        act = action / self.__class__.action_scaler
        if decision_type == DecisionType.Supply:
            action = Action(act_idx, target_idx, int(abs(act)))
        else:
            action = Action(target_idx, act_idx, int(abs(act)))
        return action
