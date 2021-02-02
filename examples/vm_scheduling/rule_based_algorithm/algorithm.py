import math

from maro.simulator.scenarios.vm_scheduling.common import Action
from maro.simulator.scenarios.vm_scheduling import AllocateAction, DecisionPayload, PostponeAction


class VMSchedulingAgent(object):
    def choose_action(self) -> Action:
        """This method will determine whether to postpone the current VM or allocate a PM to the current VM.
        """
        pass
