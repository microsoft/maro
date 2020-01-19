from maro.simulator.scenarios.ecr.common import ActionScope

class OnlineLPActionShaping():
    def __init__(self):
        pass

    def __call__(self, scope: ActionScope, model_action: int) -> int:
        # model action: num from vessel to port
        execute_action = model_action

        execute_action = min(execute_action, scope.discharge)
        execute_action = max(execute_action, -scope.load)
        execute_action = int(execute_action)

        # execute_action: num from vessel to port
        return execute_action