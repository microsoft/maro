import pdb
import random

from maro.simulator.scenarios.vm_scheduling.common import Action
from maro.simulator.scenarios.vm_scheduling import AllocateAction, DecisionPayload, PostponeAction

from algorithm import VMSchedulingAgent


class BestFit(VMSchedulingAgent):
    def __init__(
        self,
        metric_type: str = "remaining_cpu_cores"
    ):
        super().__init__()
        self._metric_type: str = metric_type

    def choose_action(self, decision_event, env) -> Action:
        env = env
        decision_event = decision_event

        valid_pm_num: int = len(decision_event.valid_pms)

        # Check whether there exists a valid PM.
        if valid_pm_num <= 0:
            # No valid PM now, postpone.
            action: PostponeAction = PostponeAction(
                vm_id=decision_event.vm_id,
                postpone_step=1
            )
        else:
            # Use a rule to choose a valid PM.
            chosen_idx: int = self._pick_pm_func(decision_event, env)
            # Take action to allocate on the chose PM.
            action: AllocateAction = AllocateAction(
                vm_id=decision_event.vm_id,
                pm_id=decision_event.valid_pms[chosen_idx]
            )

        return action

    def _pick_pm_func(self, decision_event, env) -> int:
        # Get the capacity and allocated cores from snapshot.
        valid_pm_info = env.snapshot_list["pms"][
            env.frame_index:decision_event.valid_pms:[
                "cpu_cores_capacity", "cpu_cores_allocated", "memory_capacity", "memory_allocated", "energy_consumption"
            ]
        ].reshape(-1, 5)
        # Calculate to get the remaining cpu cores.
        cpu_cores_remaining = valid_pm_info[:, 0] - valid_pm_info[:, 1]
        # Calculate to get the remaining memory.
        memory_remaining = valid_pm_info[:, 2] - valid_pm_info[:, 3]
        # Calculate to get the energy consumption.
        energy_consumption = valid_pm_info[:, 4]
        # Choose the PM with the preference rule.
        chosen_idx = 0
        if self._metric_type == 'remaining_cpu_cores':
            minimum_remaining_cpu_cores = cpu_cores_remaining[0]
            for i, remaining in enumerate(cpu_cores_remaining):
                if remaining < minimum_remaining_cpu_cores:
                    chosen_idx = i
                    minimum_remaining_cpu_cores = remaining
        elif self._metric_type == 'remaining_memory':
            minimum_remaining_memory = memory_remaining[0]
            for i, remaining in enumerate(memory_remaining):
                if remaining < minimum_remaining_memory:
                    chosen_idx = i
                    minimum_remaining_memory = remaining
        elif self._metric_type == 'energy_consumption':
            maximum_energy_consumption = energy_consumption[0]
            for i, energy in enumerate(energy_consumption):
                if energy > maximum_energy_consumption:
                    chosen_idx = i
                    maximum_energy_consumption = energy
        elif self._metric_type == 'remaining_cpu_cores_and_energy_consumption':
            maximum_energy_consumption = energy_consumption[0]
            minimum_remaining_cpu_cores = cpu_cores_remaining[0]
            for i, remaining in enumerate(cpu_cores_remaining):
                energy = energy_consumption[i]
                if remaining < minimum_remaining_cpu_cores or \
                        (remaining == minimum_remaining_cpu_cores and energy > maximum_energy_consumption):
                    chosen_idx = i
                    minimum_remaining_cpu_cores = remaining
                    maximum_energy_consumption = energy

        return chosen_idx
