# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List

from maro.event_buffer import DECISION_EVENT, Event, EventBuffer
from maro.simulator.scenarios.abs_business_engine import AbsBusinessEngine
from maro.simulator.scenarios.helpers import DocableDict

from .common import Action
from .events import DataCenterEvents
from .virtual_machine import VirtualMachine
from .physical_machine import PhysicalMachine

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
        # vm requirement event
        vm_required_evt = self._event_buffer.gen_atom_event(tick, DataCenterEvents.REQUIRE, payload=None)
        self._event_buffer.insert_event(vm_required_evt)

    def _init(self):
        # load config
        self._register_events()

        self.machine: List[PhysicalMachine] = []
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
        self._event_buffer.register_event_handler(DataCenterEvents.REQUIRE, self._on_vm_required)
        self._event_buffer.register_event_handler(DataCenterEvents.FINISHED, self._on_vm_finished)

        self._event_buffer.register_event_handler(DECISION_EVENT, self._on_action_received)

    def _on_vm_required(self, evt: Event):
        """Callback when there is a VM requirement generated."""
        # load VM data

    def _on_vm_finished(self, evt: Event):
        """Callback when there is a VM in the end cycle."""

    def _on_action_received(self, evt: Event):
        """Callback wen we get an action from agent."""
        action: Action = evt.payload

        if action is not None:
            # Load vm data

            if action.assign:
                # update vm
                # update pm resource
                pass
            else:
                pass
                # check buffer time
                # if buffer time < 1, requirement failed
                # else buffer time -= 1
