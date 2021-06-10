# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
import numpy as np

from maro.simulator import Env
from maro.simulator.scenarios.vm_scheduling.common import Action
from maro.simulator.scenarios.vm_scheduling import AllocateAction, DecisionPayload, PostponeAction

from maro.utils import Logger


class ILPAgent(object):
    def __init__(
        self,
        env: Env,
        pm_num: int,
        agent_config: dict,
        simulation_logger: Logger,
        ilp_logger: Logger,
        log_path: str
    ):
        self.env = env
        self.pm_num = pm_num

        _, _, self.is_done = self.env.step(None)

        self.ilp_agent = IlpAgent(
            ilp_config=agent_config["ilp"],
            pm_capacity=agent_config["pm_capacity"],
            vm_table_path=agent_config["vm_table"],
            env_start_tick=agent_config["start_tick"],
            env_duration=agent_config["durations"],
            simulation_logger=simulation_logger,
            ilp_logger=ilp_logger,
            log_path=log_path
        )

        self.actions = dict()

        self._init_agent()

    def _init_agent(self):
        while not self.is_done:
            # Get live VMs in each PM.
            live_vm_set_list: List[Set[int]] = [env._business_engine._machines[idx]._live_vms for idx in range(pm_num)]

            action = self.ilp_agent.choose_action(
                env_tick=env.tick,
                cur_vm_id=decision_event.vm_id,
                live_vm_set_list=live_vm_set_list,
            )

            try:
                self.actions[action.vm_id] = action.pm_id
            except:
                self.actions[action.vm_id] = self.pm_num

            _, _, self.is_done = self.env.step(action)

    def choose_action(self, decision_event: DecisionPayload) -> int:
        return self.actions[decision_event.vm_id]
