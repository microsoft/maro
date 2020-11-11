from maro.rl import ActionShaper


class DiscreteActionShaper(ActionShaper):
    """The shaping class to transform the action in [-1, 1] to actual repositioning function."""
    def __init__(self, action_dim):
        super().__init__()
        self._action_dim = action_dim
        self._zero_action = self._action_dim // 2

    def __call__(self, decision_event, model_action):
        """Shaping the action in [-1,1] range to the actual repositioning function.

        This function maps integer model action within the range of [-A, A] to actual action. We define negative actual
        action as discharge resource from vessel to port and positive action as upload from port to vessel, so the
        upper bound and lower bound of actual action are the resource in dynamic and static node respectively.

        Args:
            decision_event (Event): The decision event from the environment.
            model_action (int): Output action, range A means the half of the agent output dim.
        """
        env_action = 0
        model_action -= self._zero_action

        action_scope = decision_event.action_scope

        if model_action < 0:
            # Discharge resource from dynamic node.
            env_action = round(int(model_action) * 1.0 / self._zero_action * action_scope.load)
        elif model_action == 0:
            env_action = 0
        else:
            # Load resource to dynamic node.
            env_action = round(int(model_action) * 1.0 / self._zero_action * action_scope.discharge)
        env_action = int(env_action)

        return env_action
