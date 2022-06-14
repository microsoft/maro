# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
from rule_based_algorithm import RuleBasedAlgorithm

from maro.simulator import Env
from maro.simulator.scenarios.vm_scheduling import AllocateAction, DecisionPayload


class BestFit(RuleBasedAlgorithm):
    def __init__(self, **kwargs):
        super().__init__()
        self._metric_type: str = kwargs["metric_type"]

    def allocate_vm(self, decision_event: DecisionPayload, env: Env) -> AllocateAction:
        # Use a rule to choose a valid PM.
        chosen_idx: int = self._pick_pm_func(decision_event, env)
        # Take action to allocate on the chose PM.
        action: AllocateAction = AllocateAction(
            vm_id=decision_event.vm_id,
            pm_id=decision_event.valid_pms[chosen_idx],
        )

        return action

    def _pick_pm_func(self, decision_event, env) -> int:
        # Get the capacity and allocated cores from snapshot.
        valid_pm_info = env.snapshot_list["pms"][
            env.frame_index : decision_event.valid_pms : [
                "cpu_cores_capacity",
                "cpu_cores_allocated",
                "memory_capacity",
                "memory_allocated",
                "energy_consumption",
            ]
        ].reshape(-1, 5)
        # Calculate to get the remaining cpu cores.
        cpu_cores_remaining = valid_pm_info[:, 0] - valid_pm_info[:, 1]
        # Calculate to get the remaining memory.
        memory_remaining = valid_pm_info[:, 2] - valid_pm_info[:, 3]
        # Calculate to get the energy consumption.
        energy_consumption = valid_pm_info[:, 4]
        # Choose the PM with the preference rule.
        chosen_idx: int = 0
        if self._metric_type == "remaining_cpu_cores":
            chosen_idx = np.argmin(cpu_cores_remaining)
        elif self._metric_type == "remaining_memory":
            chosen_idx = np.argmin(memory_remaining)
        elif self._metric_type == "energy_consumption":
            chosen_idx = np.argmax(energy_consumption)
        elif self._metric_type == "remaining_cpu_cores_and_energy_consumption":
            maximum_energy_consumption = energy_consumption[0]
            minimum_remaining_cpu_cores = cpu_cores_remaining[0]
            for i, remaining in enumerate(cpu_cores_remaining):
                energy = energy_consumption[i]
                if remaining < minimum_remaining_cpu_cores or (
                    remaining == minimum_remaining_cpu_cores and energy > maximum_energy_consumption
                ):
                    chosen_idx = i
                    minimum_remaining_cpu_cores = remaining
                    maximum_energy_consumption = energy

        return chosen_idx
