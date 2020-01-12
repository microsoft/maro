from maro.simulator.scenarios.ecr.common import ActionScope

class ReverseActionShaping():
    def __init__(self, action_shaping):
        self._zero_action_index = action_shaping.zero_action_index
        self._action_space = action_shaping.action_space

    def __call__(self, scope: ActionScope, env_action: int):
        model_action = 0
        if env_action > 0:
            model_action = self._action_space.index(round(10 * env_action / scope.discharge) / 10)
        elif env_action < 0:
            # TODO: not fully match now
            model_action = self._action_space.index(round(10 * env_action / scope.load) / 10)
        return model_action

    @property
    def action_space(self):
        return self._action_space