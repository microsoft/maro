import math
from maro.rl import ActionShaper

class DiscreteActionShaper(ActionShaper):
    def __init__(self, action_dim):
        self._action_dim = action_dim
        self._zero_action = self._action_dim // 2

    def __call__(self, pending_action, model_action):
        '''
        This function maps integer model action within the range of [-A, A] to actual action. We define negative actual 
        action as discharge resource from vessel to port and positive action as upload from port to vessel, so the 
        upper bound and lower bound of actual action are the resource in dynamic and static node respectively.

        Args:
            pending_action (Event)
            model_action (int): output action, range A means the half of the agent output dim
        '''
        tick = pending_action.tick
        env_action = 0
        action_index = model_action
        model_action -= self._zero_action

        action_scope = pending_action.action_scope
        
        if model_action < 0:
            # discharge resource from dynamic node
            env_action = round(int(model_action) * 1.0/self._zero_action * action_scope.load)
        elif model_action == 0:
            env_action =0
        else:
            # load resource to dynamic node
            env_action = round(int(model_action) * 1.0/self._zero_action * action_scope.discharge)
        env_action = int(env_action)

        return env_action


