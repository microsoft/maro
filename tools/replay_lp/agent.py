# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.simulator.scenarios.ecr.common import Action, DecisionEvent

class LPAgent(object):
    def __init__(self, algorithm, action_shaping, port_idx2name, vessel_idx2name):
        self._action_shaping = action_shaping
        self._algorithm = algorithm
        self._port_idx2name = port_idx2name
        self._vessel_idx2name = vessel_idx2name
    
    def choose_action(self,
                      decision_event: DecisionEvent,
                      initial_port_empty: dict = None,
                      initial_port_on_consignee: dict = None,
                      initial_port_full: dict = None,
                      initial_vessel_empty: dict = None,
                      initial_vessel_full: dict = None,
                      ) -> Action:
        tick = decision_event.tick
        port_idx = decision_event.port_idx
        vessel_idx = decision_event.vessel_idx
        port_name = self._port_idx2name[decision_event.port_idx]
        vessel_name = self._vessel_idx2name[decision_event.vessel_idx]
                      
        model_action = self._algorithm.choose_action(current_tick=tick,
                                                     port_code=port_name,
                                                     vessel_code=vessel_name,
                                                     initial_port_empty=initial_port_empty,
                                                     initial_port_on_consignee=initial_port_on_consignee,
                                                     initial_port_full=initial_port_full,
                                                     initial_vessel_empty=initial_vessel_empty,
                                                     initial_vessel_full=initial_vessel_full
                                                     )

        action_scope = decision_event.action_scope
        early_discharge = decision_event.early_discharge
        actual_action = self._action_shaping(scope=action_scope, early_discharge=early_discharge, model_action=model_action)
        env_action = Action(vessel_idx, port_idx, actual_action)
        return env_action