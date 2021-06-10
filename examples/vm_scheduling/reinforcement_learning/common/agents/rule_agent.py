# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
import numpy as np
import importlib

from maro.simulator import Env
from maro.simulator.scenarios.vm_scheduling.common import Action
from maro.simulator.scenarios.vm_scheduling import AllocateAction, DecisionPayload, PostponeAction

from examples.vm_scheduling.rule_based_algorithm.agent import VMSchedulingAgent


def import_class(name):
    components = name.rsplit('.', 1)
    mod = importlib.import_module(components[0])
    mod = getattr(mod, components[1])
    return mod


class RuleAgent(object):
    def __init__(
        self,
        env: Env,
        pm_num: int,
        agent_config: dict
    ):
        self.env = env
        self.pm_num = pm_num

        metrics, self.decision_event, self.is_done = self.env.step(None)

        algorithm_class = import_class(agent_config["type"])
        if agent_config["args"] is None:
            algorithm = algorithm_class(env=self.env)
        else:
            algorithm = algorithm_class(env=self.env, **agent_config["args"])

        self.agent = VMSchedulingAgent(algorithm)

        self.actions = dict()

        self._init_agent()

    def _init_agent(self):
        while not self.is_done:
            action = self.agent.choose_action(self.decision_event, self.env)

            try:
                self.actions[action.vm_id] = action.pm_id
            except:
                self.actions[action.vm_id] = self.pm_num

            metrics, self.decision_event, self.is_done = self.env.step(action)

        print(metrics)

    def choose_action(self, decision_event: DecisionPayload) -> int:
        return self.actions[decision_event.vm_id]
