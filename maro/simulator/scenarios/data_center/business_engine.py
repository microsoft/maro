# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from typing import List

from yaml import safe_load

from maro.event_buffer import AtomEvent, CascadeEvent, EventBuffer, MaroEvents
from maro.simulator.scenarios.abs_business_engine import AbsBusinessEngine
from maro.simulator.scenarios.helpers import DocableDict

from .common import Action, DecisionEvent, Latency, VmFinishedPayload, VmRequirementPayload
from .events import Events
from .physical_machine import PhysicalMachine
from .virtual_machine import VirtualMachine

metrics_desc = """

energy_consumption (int): Current total energy consumption.
success_requirements (int): Accumulative successful VM requirements until now.
failed_requirements (int): Accumulative failed VM requirements until now.
total_latency (int): Accumulative buffer time until now.
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

        self._energy_consumption: int = 0
        self._success_requirements: int = 0
        self._failed_requirements: int = 0
        self._total_latency: Latency = Latency()

        # Load configurations.
        self._load_configs()
        self._register_events()

        # PM initialize.
        self._machines: List[PhysicalMachine] = []
        self._machines = [
            PhysicalMachine(
                id=i,
                cap_cpu=self._conf["pm_cap_cpu"],
                cap_mem=self._conf["pm_cap_mem"]
            ) for i in range(self._conf["pm_amount"])
        ]
        # VM initialize.
        self._vm: dict = {}

        self._tick: int = 0
        self._delay_duration: int = self._conf["delay_duration"]

    def _load_configs(self):
        """Load configurations."""
        with open(os.path.join(self._config_path, "config.yml")) as fp:
            self._conf = safe_load(fp)

    def step(self, tick: int):
        """Push business to next step.

        Args:
            tick (int): Current tick to process.
        """
        self._tick = tick
        # Load VM info into payload.

        # Update all live VMs memory utilization.
        self._update_vm_util()
        # Update all PM memory utilization.
        self._update_pm_util()
        # Generate VM requirement events from data file.
        # It might be implemented by a for loop to process VMs in each tick.
        # TODO
        vm_required_evt = self._event_buffer.gen_cascade_event(tick, Events.REQUIREMENTS, payload=None)
        self._event_buffer.insert_event(vm_required_evt)

    def get_metrics(self) -> dict:
        """Get current enviornment metrics information.

        Returns:
            dict: Metrics information.
        """

        return DocableDict(
            metrics_desc,
            energy_consumption=self._energy_consumption,
            success_requirements=self._success_requirements,
            failed_requirements=self._failed_requirements,
            total_latency=self._total_latency
        )

    def _register_events(self):
        # Register our own events and their callback handlers.
        self._event_buffer.register_event_handler(Events.REQUIREMENTS, self._on_vm_required)
        self._event_buffer.register_event_handler(Events.FINISHED, self._on_vm_finished)

        # Generate decision event.
        self._event_buffer.register_event_handler(MaroEvents.TAKE_ACTION, self._on_action_received)

    def _update_vm_util(self):
        """Update all live VMs memory utilization."""
        for _, vm in self._vm.items():
            vm.util_cpu = vm.util_series[self._tick]

    def _update_pm_util(self):
        """Update memory utilization occupied by total VMs on each PM."""
        for pm in self._machines:
            total_cpu: int = 0
            for vm_id in pm.vm_set:
                vm = self._vm[vm_id]
                cur_util_cores = vm.util_cpu * vm.req_cpu / 100
                total_cpu += cur_util_cores
            pm.util_cpu = total_cpu / pm.cap_cpu * 100
            pm.update_util_series(self._tick)

    def _on_vm_required(self, evt: CascadeEvent):
        """Callback when there is a VM requirement generated."""
        # Get VM data from payload.
        payload: VmRequirementPayload = evt.payload
        vm_req: VirtualMachine = payload.vm_req
        buffer_time: int = payload.buffer_time

        # Check all valid PMs.
        valid_pm_list = [
            pm.id
            for pm in self._machines
            if (pm.cap_cpu - (pm.cap_cpu * pm.util_cpu / 100)) >= vm_req.req_cpu
        ]

        if len(valid_pm_list) > 0:
            # Generate pending decision.
            decision_payload = DecisionEvent(
                valid_pm=valid_pm_list,
                vm_info=vm_req,
                buffer_time=buffer_time
            )
            pending_decision_evt = self._event_buffer.gen_decision_event(evt.tick, payload=decision_payload)
            evt.add_immediate_event(pending_decision_evt)
        else:
            # Postpone the buffer duration ticks by config setting.
            if buffer_time > 0:
                postpone_payload = payload
                postpone_payload.buffer_time -= self._delay_duration
                self._total_latency.latency_due_to_resource += self._delay_duration
                postpone_evt = self._event_buffer.gen_cascade_event(
                    evt.tick + self._delay_duration, Events.REQUIREMENTS, payload=postpone_payload)
                self._event_buffer.insert_event(postpone_evt)
            else:
                # Fail
                # TODO Implement failure logic.
                self._failed_requirements += 1

    def _on_vm_finished(self, evt: AtomEvent):
        """Callback when there is a VM in the end cycle."""
        # Get the end-cycle VM info.
        payload: VmFinishedPayload = evt.payload
        vm_id = payload.vm_id
        virtual_machine: VirtualMachine = self._vm[vm_id]
        util_vm_cores = virtual_machine.util_cpu * virtual_machine.req_cpu

        # Release PM resources.
        physical_machine: PhysicalMachine = self._machines[virtual_machine.pm_id]
        physical_machine.req_cpu -= virtual_machine.req_cpu
        physical_machine.req_mem -= virtual_machine.req_mem
        physical_machine.util_cpu = (
            (physical_machine.cap_cpu * physical_machine.util_cpu / 100 - util_vm_cores)
            * 100 / physical_machine.cap_cpu
        )
        physical_machine.remove_vm(vm_id)

        # Remove dead VM.
        self._vm.pop(vm_id)

        # VM allocation succeed.
        self._success_requirements += 1

    def _on_action_received(self, evt: CascadeEvent):
        """Callback wen we get an action from agent."""
        cur_tick: int = evt.tick
        action: Action = evt.payload
        assign: bool = action.assign
        virtual_machine: VirtualMachine = action.vm_req

        if assign:
            pm_id = action.pm_id
            lifetime = virtual_machine.lifetime
            cur_util_vm_cores = virtual_machine.util_series[cur_tick] * virtual_machine.req_cpu / 100
            # Update VM information.
            virtual_machine.pm_id = pm_id
            virtual_machine.util_cpu = cur_util_vm_cores / virtual_machine.req_cpu
            virtual_machine.start_tick = cur_tick
            virtual_machine.end_tick = cur_tick + lifetime
            self._vm[virtual_machine.id] = virtual_machine

            # Generate VM finished event.
            finished_payload: VmFinishedPayload = VmFinishedPayload(virtual_machine.id)
            finished_evt = self._event_buffer.gen_atom_event(cur_tick + lifetime, payload=finished_payload)
            self._event_buffer.insert_event(finished_evt)

            # Update PM resources requested by VM.
            pm = self._machines[pm_id]
            pm.add_vm(virtual_machine.id)
            pm.req_cpu += virtual_machine.req_cpu
            pm.req_mem += virtual_machine.req_mem
            pm.util_cpu = ((pm.cap_cpu * pm.util_cpu / 100) + cur_util_vm_cores) / pm.cap_cpu
        else:
            buffer_time = action.buffer_time
            # Postpone the buffer duration ticks by config setting.
            if buffer_time > 0:
                requirement_payload = VmRequirementPayload(virtual_machine, buffer_time - self._delay_duration)
                self._total_latency.latency_due_to_agent += self._delay_duration

                postpone_evt = self._event_buffer.gen_cascade_event(
                    evt.tick + self._delay_duration, payload=requirement_payload)
                self._event_buffer.insert_event(postpone_evt)
            else:
                # Fail
                # TODO Implement failure logic.
                self._failed_requirements += 1
