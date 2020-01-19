from maro.simulator.scenarios.ecr.common import Action, DecisionEvent

class OnlineLPAgent(object):
    def __init__(self, algorithm, action_shaping, port_idx2name, vessel_idx2name):
        self._action_shaping = action_shaping
        self._algorithm = algorithm
        self._port_idx2name = port_idx2name
        self._vessel_idx2name = vessel_idx2name
    
    def choose_action(self,
                      decision_event: DecisionEvent,
                      finished_events: list,
                      snapshot_list: list,
                      initial_port_empty: dict = None,
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
                                                     finished_events=finished_events,
                                                     snapshot_list=snapshot_list,
                                                     initial_port_empty=initial_port_empty,
                                                     initial_vessel_empty=initial_vessel_empty,
                                                     initial_vessel_full=initial_vessel_full
                                                     )

        action_scope = decision_event.action_scope
        actual_action = self._action_shaping(scope=action_scope, model_action=model_action)
        env_action = Action(vessel_idx, port_idx, actual_action)
        return env_action
    
    def reset(self):
        self._algorithm.reset()