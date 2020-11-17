# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from typing import Dict, List

from yaml import safe_load

from maro.event_buffer import AtomEvent, CascadeEvent, EventBuffer, MaroEvents
from maro.simulator.scenarios.abs_business_engine import AbsBusinessEngine
from maro.simulator.scenarios.helpers import DocableDict

from .common import (Action, DecisionPayload, Latency, PostponeType,
                     ValidPhysicalMachine, VmFinishedPayload,
                     VmRequirementPayload)
from .events import Events
from .physical_machine import PhysicalMachine
from .virtual_machine import VirtualMachine

metrics_desc = """

energy_consumption (int): Current total energy consumption.
success_requirements (int): Accumulative successful VM requirements until now.
failed_requirements (int): Accumulative failed VM requirements until now.
total_latency (int): Accumulative used buffer time until now.
"""


class DataCenterBusinessEngine(AbsBusinessEngine):
    def __init__(
        self, event_buffer: EventBuffer, topology: str, start_tick: int,
        max_tick: int, snapshot_resolution: int, max_snapshots: int, additional_options: dict = {}
    ):
        super().__init__(
            scenario_name="data_center", event_buffer=event_buffer, topology=topology, start_tick=start_tick,
            max_tick=max_tick, snapshot_resolution=snapshot_resolution, max_snapshots=max_snapshots,
            additional_options=additional_options
        )

        # Env metrics.
        self._energy_consumption: int = 0
        self._successful_requirements: int = 0
        self._failed_requirements: int = 0
        self._total_latency: Latency = Latency()

        # Load configurations.
        self._load_configs()
        self._register_events()

        # PMs list used for quick accessing.
        self._init_machines()
        # All living VMs.
        self._live_vm: Dict[int, VirtualMachine] = {}
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
            self._conf = safe_load(fp)

        self._delay_duration: int = self._conf["delay_duration"]
        self._pm_amount: int = self._conf["pm_amount"]
        self._pm_cap_cpu: int = self._conf["pm_cap_cpu"]
        self._pm_cap_mem: int = self._conf["pm_cap_mem"]

    def _init_machines(self):
        """Initialize the physical machines based on the config setting. The PM id starts from 0."""

        self._machines: List[PhysicalMachine] = [
            PhysicalMachine(
                id=i,
                cap_cpu=self._pm_cap_cpu,
                cap_mem=self._pm_cap_mem
            ) for i in range(self._pm_amount)
        ]

    def step(self, tick: int):
        """Push business to next step.

        Args:
            tick (int): Current tick to process.
        """
        self._tick = tick

        # Update all live VMs CPU utilization.
        self._update_vm_util()
        # Update all PM CPU utilization.
        self._update_pm_util()
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
            energy_consumption=self._energy_consumption,
            success_requirements=self._successful_requirements,
            failed_requirements=self._failed_requirements,
            total_latency=self._total_latency
        )

    def _register_events(self):
        # Register our own events and their callback handlers.
        self._event_buffer.register_event_handler(event_type=Events.REQUIREMENTS, handler=self._on_vm_required)
        self._event_buffer.register_event_handler(event_type=Events.FINISHED, handler=self._on_vm_finished)

        # Generate decision event.
        self._event_buffer.register_event_handler(event_type=MaroEvents.TAKE_ACTION, handler=self._on_action_received)

    def _update_vm_util(self):
        """Update all live VMs CPU utilization.

        The length of VMs utilization series could be difference among all VMs,
        because index 0 represents the VM's CPU utilization at the tick it starts.
        """

        for vm in self._live_vm.values():
            vm.util_cpu = vm.get_util(cur_tick=self._tick)

    def _update_pm_util(self):
        """Update CPU utilization occupied by total VMs on each PM."""

        for pm in self._machines:
            total_util_cpu: int = 0
            for vm_id in pm.vm_set:
                vm = self._live_vm[vm_id]
                total_util_cpu += vm.util_cpu * vm.req_cpu / 100
            pm.util_cpu = total_util_cpu / pm.cap_cpu * 100
            pm.update_util_series(self._tick)
            self._energy_consumption += self._cpu_util_to_energy_consumption(pm.util_cpu)

    def _cpu_util_to_energy_consumption(self, cpu_util: float) -> float:
        """Convert the CPU utilization to energy consumption."""

        power: float = self._conf["calibration parameter"]
        # NOTE: Energy comsumption parameters should refer to more research.
        busy_power = self._conf["busy_power"]
        idle_power = self._conf["idle_power"]
        return idle_power + (busy_power - idle_power) * (2 * cpu_util - pow(cpu_util, power))

    def _postpone_vm_requirement(self, postpone_type: PostponeType, vm_id: int, remaining_buffer_time: int):
        """Postpone VM requirement."""

        if remaining_buffer_time >= self._delay_duration:
            if postpone_type == PostponeType.Resource:
                self._total_latency.latency_due_to_resource += self._delay_duration
            elif postpone_type == PostponeType.Agent:
                self._total_latency.latency_due_to_agent += self._delay_duration

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
            # Add failed requirements.
            self._failed_requirements += 1

    def _get_valid_pm(self, vm_req_cpu: int) -> List[ValidPhysicalMachine]:
        """Check all valid PMs.

        There are three situations:
            1. All PMs are empty.
            2. Some PMs are empty but others are not.
            3. All PMs are not empty.
        Situation 1: Return the first empty PM info.
        Situation 2: Return all PMs with enough resources but not empty and plus the first empty PM.
        Situation 3: Return all PMs with enough resources but not empty.

        Args: vm_req_cpu (int): The CPU resource of the VM requirement.
        """
        # NOTE: Should we implement this logic inside the action scope?
        # TODO: In oversubscribable scenario, we should consider more situations, like
        #       the PM type (oversubscribable and non-oversubscribable).

        valid_pm_list = []
        for pm in self._machines:
            if pm.req_cpu == 0:
                valid_pm_list.append(
                    ValidPhysicalMachine(pm_id=pm.id, remaining_cpu=pm.cap_cpu, remaining_mem=pm.cap_mem)
                )
                break
            elif pm.req_cpu > 0 and (pm.cap_cpu - pm.req_cpu) >= vm_req_cpu:
                valid_pm_list.append(ValidPhysicalMachine(
                    pm_id=pm.id,
                    remaining_cpu=pm.cap_cpu - pm.req_cpu,
                    remaining_mem=pm.cap_mem - pm.req_mem
                ))
        return valid_pm_list

    def _on_vm_required(self, vm_requirement_event: CascadeEvent):
        """Callback when there is a VM requirement generated."""
        # Get VM data from payload.
        payload: VmRequirementPayload = vm_requirement_event.payload

        vm_req: VirtualMachine = payload.vm_req
        remaining_buffer_time: int = payload.remaining_buffer_time

        self._pending_vm_req_payload[vm_req.id] = payload

        valid_pm_list = self._get_valid_pm(vm_req.req_cpu)

        if len(valid_pm_list) > 0:
            # Generate pending decision.
            decision_payload = DecisionPayload(
                valid_pm=valid_pm_list,
                vm_id=vm_req.id,
                vm_req_cpu=vm_req.req_cpu,
                vm_req_mem=vm_req.req_mem,
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

    def _on_vm_finished(self, evt: AtomEvent):
        """Callback when there is a VM ready to be terminated."""

        # Get the VM info.
        payload: VmFinishedPayload = evt.payload
        vm_id = payload.vm_id
        vm: VirtualMachine = self._live_vm[vm_id]

        # Release PM resources.
        pm: PhysicalMachine = self._machines[vm.pm_id]
        pm.req_cpu -= vm.req_cpu
        pm.req_mem -= vm.req_mem
        pm.util_cpu = (pm.cap_cpu * pm.util_cpu - vm.req_cpu * vm.util_cpu) / pm.cap_cpu
        pm.remove_vm(vm_id)

        # Remove dead VM.
        self._live_vm.pop(vm_id)

        # VM assignment succeed.
        self._successful_requirements += 1

    def _on_action_received(self, evt: CascadeEvent):
        """Callback wen we get an action from agent."""
        cur_tick: int = evt.tick

        action: Action = evt.payload
        vm_id: int = action.vm_id
        assign: bool = action.assign

        if vm_id not in self._pending_vm_req_payload:
            print("The VM id sent by agent is invalid.")

        vm: VirtualMachine = self._pending_vm_req_payload[vm_id].vm_req

        if assign:
            pm_id = action.pm_id
            lifetime = vm.lifetime

            # Update VM information.
            vm.pm_id = pm_id
            vm.start_tick = cur_tick
            vm.end_tick = cur_tick + lifetime
            vm.util_cpu = vm.get_util(cur_tick=cur_tick)

            # Pop out the VM from pending requirements and add to live VM dict.
            self._pending_vm_req_payload.pop(vm_id)
            self._live_vm[vm_id] = vm

            # Generate VM finished event.
            finished_payload: VmFinishedPayload = VmFinishedPayload(vm.vm_id)
            finished_event = self._event_buffer.gen_atom_event(
                tick=cur_tick + lifetime,
                payload=finished_payload
            )
            self._event_buffer.insert_event(event=finished_event)

            # Update PM resources requested by VM.
            pm = self._machines[pm_id]
            pm.add_vm(vm.vm_id)
            pm.req_cpu += vm.req_cpu
            pm.req_mem += vm.req_mem
            pm.util_cpu = (pm.cap_cpu * pm.util_cpu + vm.req_cpu * vm.util_cpu) / pm.cap_cpu
        else:
            remaining_buffer_time = action.remaining_buffer_time
            # Either postpone the requirement event or failed.
            self._postpone_vm_requirement(
                postpone_type=PostponeType.Agent,
                vm_id=vm.vm_id,
                remaining_buffer_time=remaining_buffer_time
            )
