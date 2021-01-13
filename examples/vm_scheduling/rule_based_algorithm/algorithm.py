import pdb
import math
import random

from maro.simulator import Env
from maro.simulator.scenarios.vm_scheduling.common import Action
from maro.simulator.scenarios.vm_scheduling import AllocateAction, DecisionPayload, PostponeAction


class Algorithm(object):
    def __init__(self):
        self.env: Env = None
        self.decision_event: DecisionPayload = None

    def get_action(self) -> Action:
        """This method will determine whether to postpone the current VM or allocate a PM to the current VM
        """
        pass
