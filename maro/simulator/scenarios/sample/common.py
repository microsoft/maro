from typing import Callable


"""
We should define our action that from agents, 
 usually it should contains a node index to tell business engine which node should take this action
"""
class Action:
    def __init__(self, node_index: int, number: int):
        self.node_index = node_index
        self.number = number


"""
Decision event used to ask agent for an action with related information
"""
class DecisionEvent:
    def __init__(self, node_index:int, action_scope_func: Callable):
        self.node_index = node_index
        
        self._action_scope_func = action_scope_func
        self._action_scope = None


    @property
    def action_scope(self):
        # NOTE: we use a function call instead of value to make sure the agent can always get latest states
        if self._action_scope is None:
            self._action_scope = self._action_scope_func(self.node_index)
        
        return self._action_scope