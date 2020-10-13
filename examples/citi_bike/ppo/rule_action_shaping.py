
from maro.simulator.scenarios.citibike.common import Action, DecisionEvent, DecisionType
import numpy as np

class ActionShaping:
    action_scaler = 0.1
    def __init__(self):
        pass

    def __call__(self, decision_evt, choices, nums):
        station_idx = decision_evt.station_idx
        ratio_to_0 = 1
        ratio_to_2 = 0.7
        amt_ratio = 0.5
        if(decision_evt.type==DecisionType.Demand):
            #demand
            target_idx = station_idx
            act_idx = list(decision_evt.action_scope.keys())[0]
            act_amt = min(decision_evt.action_scope[target_idx],decision_evt.action_scope[act_idx]) * amt_ratio
            # if(station_idx==0):
            #     target_idx = 0
            #     act_idx = 1
            #     act_amt = int(ratio_to_0*decision_evt.action_scope[1])
            # elif(station_idx==2):
            #     target_idx = 2
            #     act_idx = 1
            #     act_amt = int(ratio_to_2*decision_evt.action_scope[1])
            # else:
            #     target_idx = 1
            #     act_idx = 1
            #     act_amt = 0
        else:
            act_idx = station_idx
            target_idx = list(decision_evt.action_scope.keys())[0]
            act_amt = min(decision_evt.action_scope[target_idx],decision_evt.action_scope[act_idx]) * amt_ratio
            # if(station_idx==1):
            #     target_idx = 0
            #     act_idx = 1
            #     act_amt = int(ratio_to_0*decision_evt.action_scope[1])
            # else:
            #     target_idx = 1
            #     act_idx = 1
            #     act_amt = 0
        action = Action(act_idx, target_idx, act_amt)
  
        return action


    def _create_action(self, act_idx, target_idx, action):
        act = action/self.__class__.action_scaler
        if act > 0:
            action = Action(act_idx, target_idx, int(abs(act)))
        else:
            action = Action(target_idx, act_idx, int(abs(act)))
        return action