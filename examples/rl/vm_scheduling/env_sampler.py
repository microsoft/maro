# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from collections import defaultdict
from os.path import dirname, realpath

import numpy as np

from maro.rl.learning import AbsEnvSampler
from maro.simulator import Env
from maro.simulator.scenarios.vm_scheduling import AllocateAction, PostponeAction

vm_path = dirname(realpath(__file__))
sys.path.insert(0, vm_path)
from config import (
    algorithm, env_conf, pm_attributes, pm_window_size, reward_shaping_conf, num_features, seed, test_env_conf,
    test_reward_shaping_conf, test_seed
)
from policies import policy_func_dict


class VMEnvSampler(AbsEnvSampler):
    def __init__(self, get_env, get_policy_func_dict, agent2policy, get_test_env=None):
        super().__init__(get_env, get_policy_func_dict, agent2policy, get_test_env=get_test_env)
        self._learn_env.set_seed(seed)
        self._test_env.set_seed(test_seed)

        # adjust the ratio of the success allocation and the total income when computing the reward
        self.num_pms = self._learn_env.business_engine._pm_amount # the number of pms
        self._durations = self._learn_env.business_engine._max_tick
        self._pm_state_history = np.zeros((pm_window_size - 1, self.num_pms, 2))
        self._legal_pm_mask = None

    def get_state(self, tick=None):
        pm_state, vm_state = self._get_pm_state(), self._get_vm_state()
        # get the legal number of PM.
        legal_pm_mask = np.zeros(self.num_pms + 1)
        if len(self.event.valid_pms) <= 0:
            # no pm available
            legal_pm_mask[self.num_pms] = 1
        else:
            legal_pm_mask[self.num_pms] = 1
            remain_cpu_dict = dict()
            for pm in self.event.valid_pms:
                # If two pms have the same remaining cpu, choose the one with the smaller id
                if pm_state[-1, pm, 0] not in remain_cpu_dict:
                    remain_cpu_dict[pm_state[-1, pm, 0]] = 1
                    legal_pm_mask[pm] = 1
                else:
                    legal_pm_mask[pm] = 0

        self._legal_pm_mask = legal_pm_mask
        return {"AGENT": np.concatenate((pm_state.flatten(), vm_state.flatten(), legal_pm_mask)).astype(np.float32)}

    def get_env_actions(self, action_info):
        action_info = action_info["AGENT"]
        model_action = action_info["action"] if isinstance(action_info, dict) else action_info
        if model_action == self.num_pms:
            return PostponeAction(vm_id=self.event.vm_id, postpone_step=1)
        else:
            return AllocateAction(vm_id=self.event.vm_id, pm_id=model_action)

    def get_reward(self, actions, tick):
        conf = reward_shaping_conf if self.env == self._learn_env else test_reward_shaping_conf
        if isinstance(actions, PostponeAction):   # postponement
            if np.sum(self._legal_pm_mask) != 1:
                reward = -0.1 * conf["alpha"] + 0.0 * conf["beta"]
            else:
                reward = 0.0 * conf["alpha"] + 0.0 * conf["beta"]
        elif self.event:
            vm_unit_price = self.env.business_engine._get_unit_price(
                self.event.vm_cpu_cores_requirement, self.event.vm_memory_requirement
            )
            reward = (
                1.0 * conf["alpha"] + conf["beta"] * vm_unit_price *
                min(self._durations - self.event.frame_index, self.event.remaining_buffer_time)
            )
        else:
            reward = .0
        return {"AGENT": np.float32(reward)}

    def _get_pm_state(self):
        total_pm_info = self.env.snapshot_list["pms"][self.env.frame_index::pm_attributes]
        total_pm_info = total_pm_info.reshape(self.num_pms, len(pm_attributes))

        # normalize the attributes of pms' cpu and memory
        self._max_cpu_capacity = np.max(total_pm_info[:, 0])
        self._max_memory_capacity = np.max(total_pm_info[:, 1])
        total_pm_info[:, 2] /= self._max_cpu_capacity
        total_pm_info[:, 3] /= self._max_memory_capacity

        # get the remaining cpu and memory of the pms
        remain_cpu = (1 - total_pm_info[:, 2]).reshape(1, self.num_pms, 1)
        remain_memory = (1 - total_pm_info[:, 3]).reshape(1, self.num_pms, 1)

        # get the pms' information
        total_pm_info = np.concatenate((remain_cpu, remain_memory), axis=2)  # (1, num_pms, 2)

        # get the sequence pms' information
        self._pm_state_history = np.concatenate((self._pm_state_history, total_pm_info), axis=0)
        return self._pm_state_history[-pm_window_size:, :, :] # (win_size, num_pms, 2)

    def _get_vm_state(self):
        return np.array([
            self.event.vm_cpu_cores_requirement / self._max_cpu_capacity,
            self.event.vm_memory_requirement / self._max_memory_capacity,
            (self._durations - self.env.tick) * 1.0 / 200,   # TODO: CHANGE 200 TO SOMETHING CONFIGURABLE
            self.env.business_engine._get_unit_price(
                self.event.vm_cpu_cores_requirement, self.event.vm_memory_requirement
            )
        ])

    def post_step(self, state, action, env_actions, reward, tick):
        self.tracker["env_metric"] = {key: metric for key, metric in self.env.metrics.items() if key != "total_latency"}
        self.tracker["env_metric"]["latency_due_to_agent"] = self.env.metrics["total_latency"].due_to_agent
        self.tracker["env_metric"]["latency_due_to_resource"] = self.env.metrics["total_latency"].due_to_resource
        if "actions_by_core_requirement" not in self.tracker:
            self.tracker["actions_by_core_requirement"] = defaultdict(list)
        if "action_sequence" not in self.tracker:
            self.tracker["action_sequence"] = []

        if self.event:
            mask = state[num_features:]
            self.tracker["actions_by_core_requirement"][self.event.vm_cpu_cores_requirement].append([action, mask])
        self.tracker["action_sequence"].append(action["action"] if isinstance(action, dict) else action)


agent2policy = {"AGENT": algorithm}

def get_env_sampler():
    return VMEnvSampler(
        get_env=lambda: Env(**env_conf),
        get_policy_func_dict=policy_func_dict,
        agent2policy=agent2policy,
        get_test_env=lambda: Env(**test_env_conf)
    )
