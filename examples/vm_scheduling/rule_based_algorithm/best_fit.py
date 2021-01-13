import pdb
import random

from maro.simulator.scenarios.vm_scheduling.common import Action
from maro.simulator.scenarios.vm_scheduling import AllocateAction, DecisionPayload, PostponeAction

from algorithm import Algorithm


class BestFit(Algorithm):
    def __init__(
        self,
        metric_type: str = "remaining_cpu_cores"
    ):
        super().__init__()
        self.metric_type: str = metric_type

    def get_action(self, decision_event, env) -> Action:
        self.env = env
        self.decision_event = decision_event

        valid_pm_num: int = len(self.decision_event.valid_pms)

        # Check whether there exists a valid PM.
        if valid_pm_num <= 0:
            # No valid PM now, postpone.
            action: PostponeAction = PostponeAction(
                vm_id=self.decision_event.vm_id,
                postpone_step=1
            )
        else:
            # Use a rule to choose a valid PM.
            chosen_idx: int = 0
            if self.metric_type == "remaining_cpu_cores":
                chosen_idx = self._pick_minimum_remaining_cpu_cores()
            elif self.metric_type == "remaining_memory":
                chosen_idx = self._pick_minimum_remaining_memory()
            elif self.metric_type == "energy_consumption":
                chosen_idx = self._pick_maximum_energy_consumption()
            elif self.metric_type == "remaining_cpu_cores_and_energy_consumption":
                chosen_idx = self._pick_minimum_remaining_cpu_cores_and_maximum_energy_consumption()
            # Take action to allocate on the chose PM.
            action: AllocateAction = AllocateAction(
                vm_id=self.decision_event.vm_id,
                pm_id=self.decision_event.valid_pms[chosen_idx]
            )

        return action

    def _pick_minimum_remaining_cpu_cores(self) -> int:
        # Get the capacity and allocated cores from snapshot.
        valid_pm_info = self.env.snapshot_list["pms"][
            self.env.frame_index:self.decision_event.valid_pms:["cpu_cores_capacity", "cpu_cores_allocated"]
        ].reshape(-1, 2)
        # Calculate to get the remaining cpu cores.
        cpu_cores_remaining = valid_pm_info[:, 0] - valid_pm_info[:, 1]
        # Choose the PM with the minimum remaining CPU.
        chosen_idx = 0
        minimum_remaining_cpu_cores = cpu_cores_remaining[0]
        for i, remaining in enumerate(cpu_cores_remaining):
            if remaining < minimum_remaining_cpu_cores:
                chosen_idx = i
                minimum_remaining_cpu_cores = remaining

        return chosen_idx

    def _pick_minimum_remaining_memory(self):
        # Get the capacity and allocated cores from snapshot.
        valid_pm_info = self.env.snapshot_list["pms"][
            self.env.frame_index:self.decision_event.valid_pms:["memory_capacity", "memory_allocated"]
        ].reshape(-1, 2)
        # Calculate to get the remaining memory.
        memory_remaining = valid_pm_info[:, 0] - valid_pm_info[:, 1]
        # Choose the PM with the minimum remaining memory.
        chosen_idx = 0
        minimum_remaining_memory = memory_remaining[0]
        for i, remaining in enumerate(memory_remaining):
            if remaining < minimum_remaining_memory:
                chosen_idx = i
                minimum_remaining_memory = remaining

        return chosen_idx

    def _pick_maximum_energy_consumption(self):
        # Get the capacity and allocated cores from snapshot.
        valid_pm_info = self.env.snapshot_list["pms"][
            self.env.frame_index:self.decision_event.valid_pms:["energy_consumption"]
        ].reshape(-1, 1)
        # Calculate to get the energy consumption.
        energy_consumption = valid_pm_info[:, 0]
        # Choose the PM with the maximum energy consumption.
        chosen_idx = 0
        maximum_energy_consumption = energy_consumption[0]
        for i, energy in enumerate(energy_consumption):
            if energy > maximum_energy_consumption:
                chosen_idx = i
                maximum_energy_consumption = energy

        return chosen_idx

    def _pick_minimum_remaining_cpu_cores_and_maximum_energy_consumption(self):
        # Get the capacity and allocated cores from snapshot.
        valid_pm_info = self.env.snapshot_list["pms"][
            self.env.frame_index:self.decision_event.valid_pms:["cpu_cores_capacity", "cpu_cores_allocated", "energy_consumption"]
        ].reshape(-1, 3)
        # Calculate to get the remaining cpu cores and energy consumption.
        cpu_cores_remaining = valid_pm_info[:, 0] - valid_pm_info[:, 1]
        energy_consumption = valid_pm_info[:, 2]
        # Choose the PM with the minimum remaining CPU and maximum energy consumption.
        chosen_idx = 0
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
