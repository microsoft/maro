# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np

from maro.rl.learning import AbsEnvWrapper
from maro.simulator import Env
from maro.simulator.scenarios.vm_scheduling import AllocateAction, PostponeAction


class VMEnvWrapper(AbsEnvWrapper):
    def __init__(
        self,
        env: Env,
        pm_attributes: list,
        vm_attributes: list,
        alpha: float,
        beta: float,
        pm_num: int,
        durations: int,
        vm_state_path: str,
        vm_window_size: int = 1,
        pm_window_size: int = 1,
        gamma: float = 0.0,
        reward_eval_delay: int = 0,
        save_replay: bool = True
    ):
        super().__init__(env, reward_eval_delay=reward_eval_delay, save_replay=save_replay, replay_agent_ids=["AGENT"])
        self._pm_attributes = pm_attributes
        self._vm_attributes = vm_attributes
        self._st = 0
        self._static_vm_states = np.load(vm_state_path)
        self._vm_window_size = vm_window_size
        self._pm_window_size = pm_window_size

        self._alpha, self._beta = alpha, beta # adjust the ratio of the success allocation and the total income when computing the reward
        self._gamma = gamma # reward discount
        self._pm_num = pm_num # the number of pms
        self._durations = durations # the duration of the whole environment

        self._pm_state_history = np.zeros((pm_window_size - 1, self._pm_num, 2))

        self._state_dim = 2 * pm_num * pm_window_size + 5 * vm_window_size
    
    @property
    def state_dim(self):
        return self._state_dim

    def get_state(self):
        pm_state, vm_state = self._get_pm_state(), self._get_vm_state()
        # get the legal number of PM.
        legal_pm_mask = np.zeros(self._pm_num + 1)
        if len(self._event.valid_pms) <= 0:
            # no pm available
            legal_pm_mask[self._pm_num] = 1
        else:
            legal_pm_mask[self._pm_num] = 1

            remain_cpu_dict = dict()
            for pm in self._event.valid_pms:
                # if two pm has same remaining cpu, only choose the one which has smaller id
                if pm_state[-1, pm, 0] not in remain_cpu_dict:
                    remain_cpu_dict[pm_state[-1, pm, 0]] = 1
                    legal_pm_mask[pm] = 1
                else:
                    legal_pm_mask[pm] = 0
        return {
            "AGENT": {
                "model": np.concatenate((pm_state.flatten(), vm_state.flatten())),
                "legal_pm_mask": legal_pm_mask
            }
        }

    def to_env_action(self, action_info):
        model_action = action_info[0] if isinstance(action_info, tuple) else action_info
        if model_action == self._pm_num:
            action = PostponeAction(vm_id=self._event.vm_id, postpone_step=1)
        else:
            action = AllocateAction(vm_id=self._event.vm_id, pm_id=model_action)
        return {"AGENT": action}

    def get_reward(self, action_info):
        model_action = action_info[0] if isinstance(action_info, tuple) else action_info
        if model_action == self._pm_num:
            if np.sum(self._state_info["AGENT"]["legal_pm"]) != 1:
                reward = -0.1 * self._alpha + 0.0 * self._beta
            else:
                reward = 0.0 * self._alpha + 0.0 * self._beta
        else:
            reward = (
                1.0 * self._alpha
                + (
                    self._event.vm_unit_price
                    * min(self._durations - self._event.frame_index, self._event.vm_lifetime)
                ) * self._beta
            )
        return {"AGENT": reward}

    def _get_pm_state(self):
        total_pm_info = self.env.snapshot_list["pms"][self.env.frame_index::self._pm_attributes]
        total_pm_info = total_pm_info.reshape(self._pm_num, len(self._pm_attributes))

        # normalize the attributes of pms' cpu and memory
        self._max_cpu_capacity = np.max(total_pm_info[:, 0])
        self._max_memory_capacity = np.max(total_pm_info[:, 1])
        total_pm_info[:, 2] /= self._max_cpu_capacity
        total_pm_info[:, 3] /= self._max_memory_capacity

        # get the remaining cpu and memory of the pms
        remain_cpu = (1 - total_pm_info[:, 2]).reshape(1, self._pm_num, 1)
        remain_memory = (1 - total_pm_info[:, 3]).reshape(1, self._pm_num, 1)

        # get the pms' information
        total_pm_info = np.concatenate((remain_cpu, remain_memory), axis=2)  # (1, pm_num, 2)

        # get the sequence pms' information
        self._pm_state_history = np.concatenate((self._pm_state_history, total_pm_info), axis=0)
        return self._pm_state_history[-self._pm_window_size:, :, :].copy() # (win_size, pm_num, 2)

    def _update_vm_state(self):
        if self._vm_window_size == 1:
            # get the vm's infomation
            vm_info = np.array([
                self._event.vm_cpu_cores_requirement,
                self._event.vm_memory_requirement,
                min(self._durations - self.env.tick, self._event.vm_lifetime) / 200,
                (self._durations - self.env.tick) * 1.0 / 200,
                self._event.vm_unit_price * min(self._durations - self.env.tick, self._event.vm_lifetime)
            ], dtype=np.float)
            vm_info[0] /= self._max_cpu_capacity
            vm_info[1] /= self._max_memory_capacity
            return vm_info
        else:
            # get the sequence vms' information
            total_vm_info = np.zeros((self._vm_window_size, len(self._vm_attributes))))

            for idx in range(self._st, self._st + self._vm_window_size):
                if idx < self._static_vm_states.shape[0]:
                    vm_info = self._static_vm_states[idx].copy()
                    vm_info[0] /= self._max_cpu_capacity
                    vm_info[1] /= self._max_memory_capacity
                    vm_info[4] = vm_info[4] * min(self._durations - vm_info[3], vm_info[2])
                    vm_info[2] = (vm_info[2] * 1.0) / 200
                    vm_info[3] = (self._durations - vm_info[3]) * 1.0 / 200
                else:
                    vm_info = np.zeros(len(self._vm_attributes), dtype=np.float)

                total_vm_info[self._vm_window_size - (idx - self._st + 1), :] = vm_info

            self._st = (self._st + 1) % self._static_vm_states.shape[0]
            return total_vm_info


env_config = {
    "basic": {
        "scenario": "vm_scheduling",
        "topology": "azure.2019.10k",
        "start_tick": 0,
        "durations": 8638,
        "snapshot_resolution": 1
    },
    "wrapper": {
        "pm_attributes": ["cpu_cores_capacity", "memory_capacity", "cpu_cores_allocated", "memory_allocated"],
        "vm_attributes": ["cpu_cores_requirement", "memory_requirement", "lifetime", "remain_time", "total_income"], 
        "alpha": 0.0,
        "beta": 1.0,
        "pm_num": 8,
        "durations": 200,
        "vm_state_path": "../data/train_vm_states.npy",
        "vm_window_size": 1,
        "pm_window_size": 1,
        "gamma": 0.9
    },
    "seed": 666
}


def get_env_wrapper():
    env = Env(**env_config["basic"])
    env.set_seed(env_config["seed"])
    return VMEnvWrapper(env, **env_config["wrapper"]) 


tmp_env_wrapper = get_env_wrapper()
AGENT_IDS = tmp_env_wrapper.agent_idx_list
STATE_DIM = tmp_env_wrapper.state_dim
del tmp_env_wrapper
