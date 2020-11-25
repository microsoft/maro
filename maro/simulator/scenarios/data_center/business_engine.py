# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from typing import Dict, List

from yaml import safe_load

from maro.event_buffer import AtomEvent, CascadeEvent, EventBuffer, MaroEvents
from maro.simulator.scenarios.abs_business_engine import AbsBusinessEngine
from maro.simulator.scenarios.helpers import DocableDict
from maro.utils.utils import DottableDict

from .common import (
    AssignAction, DecisionPayload, Latency, PostponeAction, PostponeType, ValidPhysicalMachine, VmFinishedPayload,
    VmRequirementPayload
)
from .events import Events
from .physical_machine import PhysicalMachine
from .virtual_machine import VirtualMachine

metrics_desc = """

total_energy_consumption (int): Current total energy consumption.
success_placement (int): Accumulative successful VM scheduling until now.
failed_schedulings (int): Accumulative failed VM scheduling until now.
total_latency (int): Accumulative used buffer time until now.
"""


class DataCenterBusinessEngine(AbsBusinessEngine):
    def __init__(
        self,
        event_buffer: EventBuffer,
        topology: str,
        start_tick: int,
        max_tick: int,
        snapshot_resolution: int,
        max_snapshots: int,
        additional_options: dict = {}
    ):
        super().__init__(
            scenario_name="data_center", event_buffer=event_buffer, topology=topology, start_tick=start_tick,
            max_tick=max_tick, snapshot_resolution=snapshot_resolution, max_snapshots=max_snapshots,
            additional_options=additional_options
        )

        # Env metrics.
        self._total_energy_consumption: int = 0
        self._success_placement: int = 0
        self._success_complete: int = 0
        self._failed_placement: int = 0
        self._total_latency: Latency = Latency()
        self._total_oversubscribtions: int = 0

        # Load configurations.
        self._load_configs()
        self._register_events()

        # PMs list used for quick accessing.
        self._init_pms()
        # All living VMs.
        self._live_vms: Dict[int, VirtualMachine] = {}
        # All requirement payload of the pending decision VMs.
        # NOTE: Need naming suggestestion.
        self._pending_vm_req_payload: Dict[int, VmRequirementPayload] = {}

        self._tick: int = 0
        self._pending_action_vm_id: int = -1

    def _load_configs(self):
        """Load configurations."""

        # Update self._config_path with current file path.
        self.update_config_root_path(__file__)
        with open(os.path.join(self._config_path, "config.yml")) as fp:
            self._config = safe_load(fp)

        self._delay_duration: int = self._config["delay_duration"]

        # Load pm config.
        with open(os.path.join(self._config_path, "config.pm.size.yml")) as fp:
            self._pm_size_dict = DottableDict(safe_load(fp))

        self._pm_amount: int =  self._pm_size_dict["N1"]["Amount"]
        self._pm_cpu_cores_capacity: int = self._pm_size_dict["N1"]["CPU"]
        self._pm_memory_capacity: int = self._pm_size_dict["N1"]["Memory"]

    def _init_pms(self):
        """Initialize the physical machines based on the config setting. The PM id starts from 0."""

        self._machines: List[PhysicalMachine] = [
            PhysicalMachine(
                id=i,
                cpu_cores_capacity=self._pm_cpu_cores_capacity,
                memory_capacity=self._pm_memory_capacity
            ) for i in range(self._pm_amount)
        ]

    def step(self, tick: int):
        """Push business to next step.

        Args:
            tick (int): Current tick to process.
        """
        self._tick = tick

        # Update all live VMs CPU utilization.
        self._update_vm_workload()
        # Update all PM CPU utilization.
        self._update_pm_workload()
        # TODO
        # Generate VM requirement events from data file.
        # It might be implemented by a for loop to process VMs in each tick.
        vm_requirement_event = self._event_buffer.gen_cascade_event(
            tick=tick,
            event_type=Events.REQUIREMENTS,
            payload=None
        )
        self._event_buffer.insert_event(event=vm_requirement_event)

    def get_metrics(self) -> DocableDict:
        """Get current environment metrics information.

        Returns:
            DocableDict: Metrics information.
        """

        return DocableDict(
            metrics_desc,
            total_energy_consumption=self._total_energy_consumption,
            success_requirements=self._success_placement,
            failed_requirements=self._failed_placement,
            total_latency=self._total_latency
        )

    def _register_events(self):
        # Register our own events and their callback handlers.
        self._event_buffer.register_event_handler(event_type=Events.REQUIREMENTS, handler=self._on_vm_required)
        self._event_buffer.register_event_handler(event_type=Events.FINISHED, handler=self._on_vm_finished)

        # Generate decision event.
        self._event_buffer.register_event_handler(event_type=MaroEvents.TAKE_ACTION, handler=self._on_action_received)

    def _update_vm_workload(self):
        """Update all live VMs CPU utilization.

        The length of VMs utilization series could be difference among all VMs,
        because index 0 represents the VM's CPU utilization at the tick it starts.
        """

        for vm in self._live_vms.values():
            vm.cpu_utilization = vm.get_utilization(cur_tick=self._tick)

    def _update_pm_workload(self):
        """Update CPU utilization occupied by total VMs on each PM."""
        total_energy: float = 0.0
        for pm in self._machines:
            total_pm_cpu_cores_used: float = 0.0
            for vm_id in pm.live_vms:
                vm = self._live_vms[vm_id]
                total_pm_cpu_cores_used += vm.cpu_utilization * vm.vcpu_cores_requirement / 100
            pm_cpu_utilization = total_pm_cpu_cores_used / pm.cpu_cores_capacity * 100
            pm.update_utilization(tick=self._tick, cpu_utilization=pm_cpu_utilization)
            # Update each PM's energy.
            energy_consumption = self._cpu_utilization_to_energy_consumption(pm.cpu_utilization)
            pm.update_energy(tick=self._tick, cur_energy=energy_consumption)
            total_energy += energy_consumption
        self._total_energy_consumption = total_energy

    def _cpu_utilization_to_energy_consumption(self, cpu_util: float) -> float:
        """Convert the CPU utilization to energy consumption.

        The formulation refers to https://dl.acm.org/doi/epdf/10.1145/1273440.1250665
        """

        power: float = self._config["calibration parameter"]
        # NOTE: Energy comsumption parameters should refer to more research.
        busy_power = self._config["busy_power"]
        idle_power = self._config["idle_power"]
        return idle_power + (busy_power - idle_power) * (2 * cpu_util - pow(cpu_util, power))

    def _postpone_vm_requirement(self, postpone_type: PostponeType, vm_id: int, remaining_buffer_time: int):
        """Postpone VM requirement."""

        if remaining_buffer_time >= self._delay_duration:
            if postpone_type == PostponeType.Resource:
                self._total_latency.due_to_resource += self._delay_duration
            elif postpone_type == PostponeType.Agent:
                self._total_latency.due_to_agent += self._delay_duration

            postpone_payload = self._pending_vm_req_payload[vm_id]
            postpone_payload.remaining_buffer_time -= self._delay_duration
            postpone_event = self._event_buffer.gen_cascade_event(
                tick=self._tick + self._delay_duration,
                event_type=Events.REQUIREMENTS,
                payload=postpone_payload
            )
            self._event_buffer.insert_event(event=postpone_event)
        else:
            # Fail
            # Pop out VM requirement payload.
            self._pending_vm_req_payload.pop(vm_id)
            # Add failed placement.
            self._failed_placement += 1

    def _get_valid_pm(self, vm_vcpu_cores_requirement: int) -> List[ValidPhysicalMachine]:
        """Check all valid PMs.

        There are three situations:
            1. All PMs are empty.
            2. Some PMs are empty but others are not.
            3. All PMs are not empty.
        Situation 1: Return the first empty PM info.
        Situation 2: Return all PMs with enough resources but not empty and plus the first empty PM.
        Situation 3: Return all PMs with enough resources but not empty.

        Args: vm_vcpu_cores_requirement (int): The vCPU cores requested by the VM.
        """
        # NOTE: Should we implement this logic inside the action scope?
        # TODO: In oversubscribable scenario, we should consider more situations, like
        #       the PM type (oversubscribable and non-oversubscribable).

        valid_pm_list = []
        for pm in self._machines:
            if pm.cpu_allocation == 0:
                valid_pm_list.append(
                    ValidPhysicalMachine(
                        pm_id=pm.id,
                        remaining_cpu=pm.cpu_cores_capacity,
                        remaining_mem=pm.memory_capacity
                    )
                )
                break
            elif pm.cpu_allocation > 0 and (pm.cpu_cores_capacity - pm.cpu_allocation) >= vm_vcpu_cores_requirement:
                valid_pm_list.append(ValidPhysicalMachine(
                    pm_id=pm.id,
                    remaining_cpu=pm.cpu_cores_capacity - pm.cpu_allocation,
                    remaining_mem=pm.memory_capacity - pm.memory_allocation
                ))

        return valid_pm_list

    def _on_vm_required(self, vm_requirement_event: CascadeEvent):
        """Callback when there is a VM requirement generated."""
        # Get VM data from payload.
        payload: VmRequirementPayload = vm_requirement_event.payload

        vm_req: VirtualMachine = payload.vm_req
        remaining_buffer_time: int = payload.remaining_buffer_time

        self._pending_vm_req_payload[vm_req.id] = payload

        valid_pm_list = self._get_valid_pm(vm_vcpu_cores_requirement=vm_req.vcpu_cores_requirement)

        if len(valid_pm_list) > 0:
            # Generate pending decision.
            decision_payload = DecisionPayload(
                valid_pms=valid_pm_list,
                vm_id=vm_req.id,
                vm_vcpu_cores_requirement=vm_req.vcpu_cores_requirement,
                vm_memory_requirement=vm_req.memory_requirement,
                remaining_buffer_time=remaining_buffer_time
            )
            pending_decision_event = self._event_buffer.gen_decision_event(
                tick=vm_requirement_event.tick, payload=decision_payload)
            vm_requirement_event.add_immediate_event(event=pending_decision_event)
        else:
            # Either postpone the requirement event or failed.
            self._postpone_vm_requirement(
                postpone_type=PostponeType.Resource,
                vm_id=vm_req.id,
                remaining_buffer_time=remaining_buffer_time
            )

    def _on_vm_finished(self, finish_event: AtomEvent):
        """Callback when there is a VM ready to be terminated."""

        # Get the VM info.
        payload: VmFinishedPayload = finish_event.payload
        vm_id = payload.vm_id
        vm: VirtualMachine = self._live_vms[vm_id]

        # Release PM resources.
        pm: PhysicalMachine = self._machines[vm.pm_id]
        pm.cpu_allocation -= vm.vcpu_cores_requirement
        pm.memory_allocation -= vm.memory_requirement
        # Calculate the PM's utilization.
        pm_cpu_utilization = (
            (pm.cpu_cores_capacity * pm.cpu_utilization - vm.vcpu_cores_requirement * vm.cpu_utilization) / pm.cpu_cores_capacity
        )
        pm.update_utilization(tick=finish_event.tick, cpu_utilization=pm_cpu_utilization)
        pm.remove_vm(vm_id)

        # Remove dead VM.
        self._live_vms.pop(vm_id)

        # VM placement succeed.
        self._success_placement += 1

    def _on_action_received(self, event: CascadeEvent):
        """Callback wen we get an action from agent."""
        cur_tick: int = event.tick
        action = event.payload
        vm_id: int = action.vm_id

        if vm_id not in self._pending_vm_req_payload:
            raise Exception(f"The VM id: '{vm_id}' sent by agent is invalid.")

        if type(action) == AssignAction:
            pm_id = action.pm_id
            vm: VirtualMachine = self._pending_vm_req_payload[vm_id].vm_req
            lifetime = vm.lifetime

            # Update VM information.
            vm.pm_id = pm_id
            vm.start_tick = cur_tick
            vm.end_tick = cur_tick + lifetime
            vm.cpu_utilization = vm.get_utilization(cur_tick=cur_tick)

            # Pop out the VM from pending requirements and add to live VM dict.
            self._pending_vm_req_payload.pop(vm_id)
            self._live_vms[vm_id] = vm

            # TODO: Current logic can not fulfill the oversubscription case.
            # Generate VM finished event.
            finished_payload: VmFinishedPayload = VmFinishedPayload(vm.vm_id)
            finished_event = self._event_buffer.gen_atom_event(
                tick=cur_tick + lifetime,
                payload=finished_payload
            )
            self._event_buffer.insert_event(event=finished_event)

            # Update PM resources requested by VM.
            pm = self._machines[pm_id]
            pm.place_vm(vm.vm_id)
            pm.cpu_allocation += vm.vcpu_cores_requirement
            pm.memory_allocation += vm.memory_requirement
            # Calculate the PM's utilization.
            pm_cpu_utilization = (
                (pm.cpu_cores_capacity * pm.cpu_utilization + vm.vcpu_cores_requirement * vm.cpu_utilization) / pm.cpu_cores_capacity
            )
            pm.update_utilization(tick=cur_tick, cpu_utilization=pm_cpu_utilization)
        elif type(action) == PostponeAction:
            remaining_buffer_time = action.remaining_buffer_time
            # Either postpone the requirement event or failed.
            self._postpone_vm_requirement(
                postpone_type=PostponeType.Agent,
                vm_id=vm_id,
                remaining_buffer_time=remaining_buffer_time
            )
