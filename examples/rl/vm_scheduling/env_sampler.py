# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from collections import defaultdict
from os.path import dirname, realpath
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from maro.rl.policy import RLPolicy
from maro.rl.rollout import AbsEnvSampler, CacheElement
from maro.simulator import Env
from maro.simulator.scenarios.vm_scheduling import AllocateAction, DecisionPayload, PostponeAction

vm_path = dirname(realpath(__file__))
sys.path.insert(0, vm_path)
from config import (
    algorithm, env_conf, num_features, pm_attributes, pm_window_size, reward_shaping_conf, seed, test_env_conf,
    test_reward_shaping_conf, test_seed
)


class VMEnvSampler(AbsEnvSampler):
    def __init__(self, get_env, policy_creator, agent2policy, get_test_env=None):
        super().__init__(get_env, policy_creator, agent2policy, get_test_env=get_test_env)
        self._learn_env.set_seed(seed)
        self._test_env.set_seed(test_seed)

        # adjust the ratio of the success allocation and the total income when computing the reward
        self.num_pms = self._learn_env.business_engine._pm_amount # the number of pms
        self._durations = self._learn_env.business_engine._max_tick
        self._pm_state_history = np.zeros((pm_window_size - 1, self.num_pms, 2))
        self._legal_pm_mask = None

    def _get_global_and_agent_state(
        self, event: DecisionPayload, tick: int = None
    ) -> Tuple[Optional[np.ndarray], Dict[Any, np.ndarray]]:
        pm_state, vm_state = self._get_pm_state(), self._get_vm_state(event)
        # get the legal number of PM.
        legal_pm_mask = np.zeros(self.num_pms + 1)
        if len(event.valid_pms) <= 0:
            # no pm available
            legal_pm_mask[self.num_pms] = 1
        else:
            legal_pm_mask[self.num_pms] = 1
            remain_cpu_dict = dict()
            for pm in event.valid_pms:
                # If two pms have the same remaining cpu, choose the one with the smaller id
                if pm_state[-1, pm, 0] not in remain_cpu_dict:
                    remain_cpu_dict[pm_state[-1, pm, 0]] = 1
                    legal_pm_mask[pm] = 1
                else:
                    legal_pm_mask[pm] = 0

        self._legal_pm_mask = legal_pm_mask
        state = np.concatenate((pm_state.flatten(), vm_state.flatten(), legal_pm_mask)).astype(np.float32)
        return None, {"AGENT": state}

    def _translate_to_env_action(self, action_dict: Dict[Any, np.ndarray], event: DecisionPayload) -> Dict[Any, object]:
        if action_dict["AGENT"] == self.num_pms:
            return {"AGENT": PostponeAction(vm_id=event.vm_id, postpone_step=1)}
        else:
            return {"AGENT": AllocateAction(vm_id=event.vm_id, pm_id=action_dict["AGENT"][0])}

    def _get_reward(self, env_action_dict: Dict[Any, object], event: DecisionPayload, tick: int) -> Dict[Any, float]:
        action = env_action_dict["AGENT"]
        conf = reward_shaping_conf if self._env == self._learn_env else test_reward_shaping_conf
        if isinstance(action, PostponeAction):   # postponement
            if np.sum(self._legal_pm_mask) != 1:
                reward = -0.1 * conf["alpha"] + 0.0 * conf["beta"]
            else:
                reward = 0.0 * conf["alpha"] + 0.0 * conf["beta"]
        else:
            reward = self._get_allocation_reward(event, conf["alpha"], conf["beta"]) if event else .0
        return {"AGENT": np.float32(reward)}

    def _get_pm_state(self):
        total_pm_info = self._env.snapshot_list["pms"][self._env.frame_index::pm_attributes]
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

    def _get_vm_state(self, event):
        return np.array([
            event.vm_cpu_cores_requirement / self._max_cpu_capacity,
            event.vm_memory_requirement / self._max_memory_capacity,
            (self._durations - self._env.tick) * 1.0 / 200,   # TODO: CHANGE 200 TO SOMETHING CONFIGURABLE
            self._env.business_engine._get_unit_price(event.vm_cpu_cores_requirement, event.vm_memory_requirement)
        ])

    def _get_allocation_reward(self, event: DecisionPayload, alpha: float, beta: float):
        vm_unit_price = self._env.business_engine._get_unit_price(
            event.vm_cpu_cores_requirement, event.vm_memory_requirement
        )
        return (alpha + beta * vm_unit_price * min(self._durations - event.frame_index, event.remaining_buffer_time))

    def _post_step(self, cache_element: CacheElement, reward: Dict[Any, float]):
        self._tracker["env_metric"] = {k: v for k, v in self._env.metrics.items() if k != "total_latency"}
        self._tracker["env_metric"]["latency_due_to_agent"] = self._env.metrics["total_latency"].due_to_agent
        self._tracker["env_metric"]["latency_due_to_resource"] = self._env.metrics["total_latency"].due_to_resource
        if "actions_by_core_requirement" not in self._tracker:
            self._tracker["actions_by_core_requirement"] = defaultdict(list)
        if "action_sequence" not in self._tracker:
            self._tracker["action_sequence"] = []

        action = cache_element.action_dict["AGENT"]
        if cache_element.state:
            mask = cache_element.state[num_features:]
            self._tracker["actions_by_core_requirement"][cache_element.event.vm_cpu_cores_requirement].append([action, mask])
        self._tracker["action_sequence"].append(action)


agent2policy = {"AGENT": f"{algorithm}.policy"}

def env_sampler_creator(policy_creator: Dict[str, Callable[[str], RLPolicy]]) -> VMEnvSampler:
    return VMEnvSampler(
        get_env=lambda: Env(**env_conf),
        policy_creator=policy_creator,
        agent2policy=agent2policy,
        get_test_env=lambda: Env(**test_env_conf)
    )
