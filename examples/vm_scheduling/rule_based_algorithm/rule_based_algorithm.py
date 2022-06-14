# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc

from maro.simulator import Env
from maro.simulator.scenarios.vm_scheduling import AllocateAction, DecisionPayload


class RuleBasedAlgorithm(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def allocate_vm(self, decision_event: DecisionPayload, env: Env) -> AllocateAction:
        """This method will determine allocate which PM to the current VM."""
        raise NotImplementedError
