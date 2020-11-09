# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from copy import deepcopy
from typing import List

from yaml import safe_load

from maro.event_buffer import AtomEvent, CascadeEvent, EventBuffer, MaroEvents
from maro.simulator.scenarios.abs_business_engine import AbsBusinessEngine
from maro.simulator.scenarios.helpers import DocableDict

from .common import Action, Latency, Payload
from .events import Events
from .physical_machine import PhysicalMachine
from .virtual_machine import VirtualMachine

metrics_desc = """

energy_consumption (int): Current total energy consumption.
success_requirements (int): Accumulative successful VM requirements until now.
total_latency (int): Accumulative buffer time until now.
"""


class DataCenterBusinessEngine(AbsBusinessEngine):
    def __init__(
        self, event_buffer: EventBuffer, topology: str, start_tick: int,
        max_tick: int, snapshot_resolution: int, max_snapshots: int, additional_options: dict = {}
    ):
        super().__init__(
            "data_center", event_buffer, topology, start_tick, max_tick,
            snapshot_resolution, max_snapshots, additional_options
        )

        self._energy_consumption: int = 0
        self._success_requirements: int = 0
        self._total_latency: Latency = Latency()

        # Load configurations.
        self._load_configs()
        self._register_events()

        # PM initialize.
        self.machines: List[PhysicalMachine] = []
        self.machines = [PhysicalMachine(id=i, cap_cpu=self._conf["pm_cap_cpu"], cap_mem=self._conf["pm_cap_mem"])
                         for i in range(self._conf["pm_amount"])]
        # VM initialize.
        self.vm: dict = {}

        self._tick: int = 0

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
        # Generate VM requirement event.
        vm_required_evt = self._event_buffer.gen_cascade_event(tick, Events.REQUIREMENTS, payload=None)
        self._event_buffer.insert_event(vm_required_evt)

    def get_metrics(self) -> dict:
        """Get current enviornment metrics information.

        Returns:
            dict: Metrics information.
        """

        energy_consumption = self._energy_consumption
        success_requirements = self._success_requirements
        total_latency = self._total_latency
        return DocableDict(
            metrics_desc,
            energy_consumption=energy_consumption,
            success_requirements=success_requirements,
            total_latency=total_latency
        )

    def _register_events(self):
        # Register our own events and their callback handlers.
        self._event_buffer.register_event_handler(Events.REQUIREMENTS, self._on_vm_required)
        self._event_buffer.register_event_handler(Events.FINISHED, self._on_vm_finished)

        # Generate decision event.
        self._event_buffer.register_event_handler(MaroEvents.TAKE_ACTION, self._on_action_received)

    def _update_vm_util(self):
        """Update all live VMs memory utilization."""
        for _, vm in self.vm.items():
            vm.util_mem = vm.show_util_series()[self._tick]

    def _update_pm_util(self):
        """Update memory utilization occupied by total VMs on each PM."""
        for pm in self.machines:
            cur_total_mem: int = 0
            for vm_id in pm.show_vm_set():
                vm = self.vm[vm_id]
                vm.util_mem = vm.show_util_series()[self._tick]
                cur_total_mem += vm.util_mem
            pm.util_mem = cur_total_mem
            pm.update_util_series(cur_total_mem)

    def _on_vm_required(self, evt: CascadeEvent):
        """Callback when there is a VM requirement generated."""
        # Get VM data from payload.
        payload: Payload = evt.payload
        vm_req: VirtualMachine = payload.vm_info
        buffer_time: int = payload.buffer_time

        if any([(pm.cap_mem - pm.util_mem) >= vm_req.req_mem for pm in self.machines]):
            # Generate pending decision.
            pending_decision_evt = self._event_buffer.gen_decision_event(evt.tick, payload=payload)
            evt.add_immediate_event(pending_decision_evt)
        else:
            # Postpone to next tick.
            if buffer_time > 0:
                postpone_payload = deepcopy(payload)
                postpone_payload.buffer_time -= 1
                self._total_latency.resource_buffer_time += 1
                postpone_evt = self._event_buffer.gen_cascade_event(evt.tick + 1, payload=postpone_payload)
                self._event_buffer.insert_event(postpone_evt)
            else:
                # Fail
                pass

    def _on_vm_finished(self, evt: AtomEvent):
        """Callback when there is a VM in the end cycle."""
        # Remove dead VMs.
        vm_id: int = evt.payload
        virtual_machine: VirtualMachine = self.vm[vm_id]
        req_cpu = virtual_machine.req_cpu
        req_mem = virtual_machine.req_mem
        util_mem = virtual_machine.util_mem
        self.vm.pop(vm_id)
        # Release PM resources.
        physical_machine: PhysicalMachine = self.machines[virtual_machine.pm_id]
        physical_machine.req_cpu -= req_cpu
        physical_machine.req_mem -= req_mem
        physical_machine.util_mem -= util_mem
        physical_machine.remove_vm(vm_id)

    def _on_action_received(self, evt: CascadeEvent):
        """Callback wen we get an action from agent."""
        cur_tick: int = evt.tick
        action: Action = evt.payload
        assign: bool = action.assign
        virtual_machine: VirtualMachine = deepcopy(action.vm_req)

        if assign:
            pm_id = action.pm_id
            vm_id = virtual_machine.id
            req_cpu = virtual_machine.req_cpu
            req_mem = virtual_machine.req_mem
            lifetime = virtual_machine.lifetime
            cur_util_mem = virtual_machine.show_util_series()[cur_tick]
            # Update VM information.
            virtual_machine.pm_id = pm_id
            virtual_machine.util_mem = cur_util_mem
            virtual_machine.start_tick = cur_tick
            virtual_machine.end_tick = cur_tick + lifetime
            self.vm[virtual_machine.id] = virtual_machine

            # Generate VM finished event.
            finished_evt = self._event_buffer.gen_atom_event(cur_tick + lifetime, payload=vm_id)
            self._event_buffer.insert_event(finished_evt)

            # Update PM resources requested by VM.
            pm = self.machines[pm_id]
            pm.add_vm(virtual_machine.id)
            pm.req_cpu += req_cpu
            pm.req_mem += req_mem
            pm.util_mem += cur_util_mem
            pm.add_vm(vm_id)
            self._success_requirements += 1
        else:
            buffer_time = action.buffer_time
            # Postpone to next tick.
            if buffer_time > 0:
                requirement_payload = Payload(action.vm_req, buffer_time - 1)
                self._total_latency.algorithm_buffer_time += 1

                postpone_evt = self._event_buffer.gen_cascade_event(evt.tick + 1, payload=requirement_payload)
                self._event_buffer.insert_event(postpone_evt)
            else:
                # Fail
                pass
