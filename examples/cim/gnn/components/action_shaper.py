from maro.rl import Shaper
from maro.simulator.scenarios.cim.common import Action, ActionType


class DiscreteActionShaper(Shaper):
    """The shaping class to transform the action in [-1, 1] to actual repositioning function."""
    def __init__(self, action_dim):
        super().__init__()
        self._action_dim = action_dim
        self._zero_action = self._action_dim // 2

    def __call__(self, model_action, decision_event):
        """Shaping the action in [-1,1] range to the actual repositioning function.

        This function maps integer model action within the range of [-A, A] to actual action. We define negative actual
        action as discharge resource from vessel to port and positive action as upload from port to vessel, so the
        upper bound and lower bound of actual action are the resource in dynamic and static node respectively.

        Args:
            model_action (int): Output action, range A means the half of the agent output dim.
            decision_event (Event): The decision event from the environment.
        """
        actual_action = 0
        model_action -= self._zero_action

        action_scope = decision_event.action_scope

        if model_action < 0:
            # Load resource to dynamic node.
            action_type = ActionType.LOAD
            actual_action = round(int(model_action) * 1.0 / self._zero_action * action_scope.load)
        elif model_action == 0:
            action_type = None
            actual_action = 0
        else:
            # Discharge resource from dynamic node.
            action_type = ActionType.DISCHARGE
            actual_action = round(int(model_action) * 1.0 / self._zero_action * action_scope.discharge)
        
        return Action(decision_event.vessel_idx, decision_event.port_idx, int(actual_action), action_type)
