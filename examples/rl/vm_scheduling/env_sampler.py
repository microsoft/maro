# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np

from maro.rl.learning import AbsEnvSampler
from maro.simulator import Env
from maro.simulator.scenarios.vm_scheduling import AllocateAction, PostponeAction


def post_step(env: Env, tracker: dict, transition):
    tracker["env_metric"] = env.metrics
    if "vm_cpu_cores_requirement" not in tracker:
        tracker["vm_core_requirement"] = []
    if "action_sequence" not in tracker:
        tracker["action_sequence"] = []

    tracker["vm_core_requirement"].append([transition.action["AGENT"], transition.state["AGENT"]["mask"]])
    tracker["action_sequence"].append(transition.action["AGENT"])


class VMEnvSampler(AbsEnvSampler):
    def __init__(
        self,
        env: Env,
        pm_attributes: list,
        vm_attributes: list,
        alpha: float,
        beta: float,
        pm_window_size: int = 1,
        gamma: float = 0.0,
        reward_eval_delay: int = 0
    ):
        super().__init__(env, reward_eval_delay=reward_eval_delay, replay_agent_ids=["AGENT"], post_step=post_step)
        self._pm_attributes = pm_attributes
        self._vm_attributes = vm_attributes
        self._st = 0
        self._pm_window_size = pm_window_size
        # adjust the ratio of the success allocation and the total income when computing the reward
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma  # reward discount
        self._num_pms = self.env.business_engine._pm_amount # the number of pms
        self._durations = self.env.business_engine._max_tick
        self._pm_state_history = np.zeros((pm_window_size - 1, self._num_pms, 2))
        self._legal_pm_mask = None
        self._state_dim = 2 * self._num_pms * pm_window_size + 4

    @property
    def state_dim(self):
        return self._state_dim

    @property
    def num_pms(self):
        return self._num_pms

    def get_state(self, tick=None):
        pm_state, vm_state = self._get_pm_state(), self._get_vm_state()
        # get the legal number of PM.
        legal_pm_mask = np.zeros(self._num_pms + 1)
        if len(self._event.valid_pms) <= 0:
            # no pm available
            legal_pm_mask[self._num_pms] = 1
        else:
            legal_pm_mask[self._num_pms] = 1
            remain_cpu_dict = dict()
            for pm in self._event.valid_pms:
                # if two pm has same remaining cpu, only choose the one which has smaller id
                if pm_state[-1, pm, 0] not in remain_cpu_dict:
                    remain_cpu_dict[pm_state[-1, pm, 0]] = 1
                    legal_pm_mask[pm] = 1
                else:
                    legal_pm_mask[pm] = 0

        self._legal_pm_mask = legal_pm_mask
        return {"AGENT": {"model": np.concatenate((pm_state.flatten(), vm_state.flatten())), "mask": legal_pm_mask}}

    def to_env_action(self, action_info):
        action_info = action_info["AGENT"]
        model_action = action_info[0] if isinstance(action_info, tuple) else action_info
        if model_action == self._num_pms:
            return PostponeAction(vm_id=self._event.vm_id, postpone_step=1)
        else:
            return AllocateAction(vm_id=self._event.vm_id, pm_id=model_action)

    def get_reward(self, actions, tick=None):
        if isinstance(actions, PostponeAction):   # postponement
            if np.sum(self._legal_pm_mask) != 1:
                reward = -0.1 * self._alpha + 0.0 * self._beta
            else:
                reward = 0.0 * self._alpha + 0.0 * self._beta
        elif self._event:
            vm_unit_price = self.env.business_engine._get_unit_price(
                self._event.vm_cpu_cores_requirement, self._event.vm_memory_requirement
            )
            reward = (
                1.0 * self._alpha + self._beta * vm_unit_price *
                min(self._durations - self._event.frame_index, self._event.remaining_buffer_time)
            )
        else:
            reward = .0
        return {"AGENT": np.float32(reward)}

    def _get_pm_state(self):
        total_pm_info = self.env.snapshot_list["pms"][self.env.frame_index::self._pm_attributes]
        total_pm_info = total_pm_info.reshape(self._num_pms, len(self._pm_attributes))

        # normalize the attributes of pms' cpu and memory
        self._max_cpu_capacity = np.max(total_pm_info[:, 0])
        self._max_memory_capacity = np.max(total_pm_info[:, 1])
        total_pm_info[:, 2] /= self._max_cpu_capacity
        total_pm_info[:, 3] /= self._max_memory_capacity

        # get the remaining cpu and memory of the pms
        remain_cpu = (1 - total_pm_info[:, 2]).reshape(1, self._num_pms, 1)
        remain_memory = (1 - total_pm_info[:, 3]).reshape(1, self._num_pms, 1)

        # get the pms' information
        total_pm_info = np.concatenate((remain_cpu, remain_memory), axis=2)  # (1, num_pms, 2)

        # get the sequence pms' information
        self._pm_state_history = np.concatenate((self._pm_state_history, total_pm_info), axis=0)
        return self._pm_state_history[-self._pm_window_size:, :, :].astype(np.float32) # (win_size, num_pms, 2)

    def _get_vm_state(self):
        vm_info = np.array([
            self._event.vm_cpu_cores_requirement / self._max_cpu_capacity,
            self._event.vm_memory_requirement / self._max_memory_capacity,
            (self._durations - self.env.tick) * 1.0 / 200,   # TODO: CHANGE 200 TO SOMETHING CONFIGURABLE
            self.env.business_engine._get_unit_price(
                self._event.vm_cpu_cores_requirement, self._event.vm_memory_requirement
            )
        ], dtype=np.float32)
        return vm_info


env_config = {
    "basic": {
        "scenario": "vm_scheduling",
        "topology": "azure.2019.10k",
        "start_tick": 0,
        "durations": 300,  # 8638
        "snapshot_resolution": 1
    },
    "wrapper": {
        "pm_attributes": ["cpu_cores_capacity", "memory_capacity", "cpu_cores_allocated", "memory_allocated"],
        "vm_attributes": ["cpu_cores_requirement", "memory_requirement", "lifetime", "remain_time", "total_income"],
        "alpha": 0.0,
        "beta": 1.0,
        "pm_window_size": 1,
        "gamma": 0.9
    },
    "seed": 666
}


eval_env_config = {
    "basic": {
        "scenario": "vm_scheduling",
        "topology": "azure.2019.10k.oversubscription",
        "start_tick": 0,
        "durations": 300,
        "snapshot_resolution": 1
    },
    "wrapper": {
        "pm_attributes": ["cpu_cores_capacity", "memory_capacity", "cpu_cores_allocated", "memory_allocated"],
        "vm_attributes": ["cpu_cores_requirement", "memory_requirement", "lifetime", "remain_time", "total_income"],
        "alpha": 0.0,
        "beta": 1.0,
        "pm_window_size": 1,
        "gamma": 0.9
    },
    "seed": 1024
}


def get_env_wrapper(replay_agent_ids=None):
    env = Env(**env_config["basic"])
    env.set_seed(env_config["seed"])
    return VMEnvWrapper(env, **env_config["wrapper"])


def get_eval_env_wrapper():
    eval_env = Env(**eval_env_config["basic"])
    eval_env.set_seed(eval_env_config["seed"])
    return VMEnvWrapper(eval_env, **eval_env_config["wrapper"])


tmp_env_wrapper = get_env_wrapper()
STATE_DIM = tmp_env_wrapper.state_dim
NUM_PMS = tmp_env_wrapper.num_pms
del tmp_env_wrapper