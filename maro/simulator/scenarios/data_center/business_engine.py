# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from copy import deepcopy
from typing import List

from maro.event_buffer import AtomEvent, CascadeEvent, EventBuffer, MaroEvents
from maro.simulator.scenarios.abs_business_engine import AbsBusinessEngine
from maro.simulator.scenarios.helpers import DocableDict

<<<<<<< HEAD
from .common import Action, RequirementPayload
from .events import Events
from .virtual_machine import VirtualMachine
=======
from .common import Action
from .events import DataCenterEvents
>>>>>>> parent of b0b6d2d... Merge from v0.2
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
        max_tick: int, snapshot_resolution: int, max_snapshots: int, additional_options: dict = None
    ):
        super().__init__(
            "data_center", event_buffer, topology, start_tick, max_tick,
            snapshot_resolution, max_snapshots, additional_options
        )

        self._energy_consumption: int = 0
        self._success_requirements: int = 0
        self._total_latency: int = 0

        self._init()

    def step(self, tick: int):
        """
        Push business to next step.

        Args:
            tick (int): Current tick
        """
        # Load VM info into payload

        # vm requirement event
        vm_required_evt = self._event_buffer.gen_cascade_event(tick, Events.REQUIREMENTS, payload=None)
        self._event_buffer.insert_event(vm_required_evt)

    def _init(self):
        # load config
        self._register_events()

        self.machines: List[PhysicalMachine] = []
        self.vm: dict = {}

    def get_metrics(self) -> dict:
        """Get current enviornment metrics information.

        Args:
            dict: A dict that contains the environment metrics.

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

        # Decision event
        self._event_buffer.register_event_handler(MaroEvents.TAKE_ACTION, self._on_action_received)

    def _on_vm_required(self, evt: CascadeEvent):
        """Callback when there is a VM requirement generated."""
        # Load VM data
        payload: RequirementPayload = evt.payload
        vm_req: VirtualMachine = payload.vm_req
        buffer_time: int = payload.buffer_time

        if any([(pm.cap_mem - pm.util_mem) >= vm_req.req_mem for pm in self.machines]):
            # Generate pending decision
            pending_decision_evt = self._event_buffer.gen_decision_event(evt.tick, payload=payload)
            evt.add_immediate_event(pending_decision_evt)
        else:
            # Postpone to next tick
            if buffer_time > 0:
                postpone_payload = deepcopy(payload)
                postpone_payload.buffer_time -= 1
                postpone_evt = self._event_buffer.gen_cascade_event(evt.tick + 1, payload=postpone_payload)
                self._event_buffer.insert_event(postpone_evt)
            else:
                # Fail
                pass

    def _on_vm_finished(self, evt: AtomEvent):
        """Callback when there is a VM in the end cycle."""

    def _on_action_received(self, evt: CascadeEvent):
        """Callback wen we get an action from agent."""
        cur_tick: int = evt.tick
        action: Action = evt.payload
        virtual_machine: VirtualMachine = deepcopy(action.vm_req)
        assign: bool = action.assign

        if assign:
            pm_id = action.pm_id
            cur_util_cpu, cur_util_mem = virtual_machine.util_series[cur_tick]
            lifetime = virtual_machine.lifetime
            # update vm
            virtual_machine.pm_id = pm_id
            virtual_machine.util_cpu = cur_util_cpu
            virtual_machine.util_mem = cur_util_mem
            virtual_machine.start_tick = cur_tick
            virtual_machine.end_tick = cur_tick + lifetime
            self.vm[virtual_machine.id] = virtual_machine
            # update PM resource requested by VM
            pm = self.machines[pm_id]
            pm.add_vm(virtual_machine.id)
            pm.req_cpu += virtual_machine.req_cpu
            pm.req_mem += virtual_machine.req_mem
        else:
            buffer_time = action.buffer_time
            # Postpone to next tick
            if buffer_time > 0:
                postpone_payload = deepcopy(action)
                postpone_payload.buffer_time -= 1
                postpone_evt = self._event_buffer.gen_cascade_event(evt.tick + 1, payload=postpone_payload)
                self._event_buffer.insert_event(postpone_evt)
            else:
                # Fail
                pass

            pass
            # check buffer time
            # if buffer time < 1, requirement failed
            # else buffer time -= 1
