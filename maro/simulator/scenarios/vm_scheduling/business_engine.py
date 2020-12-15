# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from typing import Dict, List

from yaml import safe_load

from maro.backends.frame import FrameBase, SnapshotList
from maro.data_lib import BinaryReader
from maro.event_buffer import AtomEvent, CascadeEvent, EventBuffer, MaroEvents
from maro.simulator.scenarios.abs_business_engine import AbsBusinessEngine
from maro.simulator.scenarios.helpers import DocableDict
from maro.utils.utils import DottableDict

from .common import (
    AssignAction, DecisionPayload, Latency, PostponeAction, PostponeType, ValidPhysicalMachine, VmFinishedPayload,
    VmRequestPayload
)
from .cpu_reader import CpuReader
from .events import Events
from .frame_builder import build_frame
from .physical_machine import PhysicalMachine
from .virtual_machine import VirtualMachine

metrics_desc = """

total_vm_requirements (int): Total VM requirements.
total_energy_consumption (float): Accumulative total energy consumption.
success_allocation (int): Accumulative successful VM allocation until now.
success_placement (int): Accumulative successful VM placement.
failed_placement (int): Accumulative failed VM placement until now.
total_latency (int): Accumulative used buffer time until now.
"""


class VmSchedulingBusinessEngine(AbsBusinessEngine):
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
            scenario_name="vm_scheduling", event_buffer=event_buffer, topology=topology, start_tick=start_tick,
            max_tick=max_tick, snapshot_resolution=snapshot_resolution, max_snapshots=max_snapshots,
            additional_options=additional_options
        )

        # Env metrics.
        self._total_vm_requests: int = 0
        self._total_energy_consumption: float = 0
        self._successful_allocation: int = 0
        self._successful_placement: int = 0
        self._failed_placement: int = 0
        self._total_latency: Latency = Latency()
        self._total_oversubscribtions: int = 0

        # Load configurations.
        self._load_configs()
        self._register_events()

        self._init_frame()

        # PMs list used for quick accessing.
        self._init_pms()
        # All living VMs.
        self._live_vms: Dict[int, VirtualMachine] = {}
        # All requirement payload of the pending decision VMs.
        # NOTE: Need naming suggestestion.
        self._pending_vm_req_payload: Dict[int, VmRequestPayload] = {}
        # All vm's cpu utilization at current tick.
        self._cpu_utilization_dict: Dict[int, float] = {}

        self._vm_reader = BinaryReader(self._config.VM_TABLE)
        self._vm_item_picker = self._vm_reader.items_tick_picker(self._start_tick, self._max_tick, time_unit="s")

        self._cpu_reader = CpuReader(self._config.CPU_READINGS)

        self._tick: int = 0

    @property
    def configs(self) -> dict:
        """dict: Current configuration."""
        return self._config

    @property
    def frame(self) -> FrameBase:
        """FrameBase: Current frame."""
        return self._frame

    @property
    def snapshots(self) -> SnapshotList:
        """SnapshotList: Current snapshot list."""
        return self._snapshots

    def _load_configs(self):
        """Load configurations."""
        # Update self._config_path with current file path.
        self.update_config_root_path(__file__)
        with open(os.path.join(self._config_path, "config.yml")) as fp:
            self._config = DottableDict(safe_load(fp))

        self._delay_duration: int = self._config.DELAY_DURATION
        self._buffer_time_budget: int = self._config.BUFFER_TIME_BUDGET
        self._pm_amount: int = self._config.PM["P1"]["AMOUNT"]

    def _init_pms(self):
        """Initialize the physical machines based on the config setting. The PM id starts from 0."""
        self._pm_cpu_cores_capacity: int = self._config.PM["P1"]["CPU"]
        self._pm_memory_capacity: int = self._config.PM["P1"]["MEMORY"]

        # TODO: Improve the scalability. Like the use of multiple PM sets.
        self._machines = self._frame.pms
        for pm_id in range(self._pm_amount):
            pm = self._machines[pm_id]
            pm.set_init_state(
                id=pm_id,
                cpu_cores_capacity=self._pm_cpu_cores_capacity,
                memory_capacity=self._pm_memory_capacity
            )

    def reset(self):
        """Reset internal states for episode."""
        self._total_energy_consumption: float = 0.0
        self._successful_allocation: int = 0
        self._successful_placement: int = 0
        self._failed_placement: int = 0
        self._total_latency: Latency = Latency()
        self._total_oversubscribtions: int = 0

        for pm in self._machines:
            pm.reset()

        self._live_vms: Dict[int, VirtualMachine] = {}
        self._pending_vm_req_payload: Dict[int, VmRequestPayload] = {}

        self._frame.reset()

        self._snapshots.reset()

        self._vm_reader.reset()
        self._vm_item_picker = self._vm_reader.items_tick_picker(self._start_tick, self._max_tick, time_unit="s")

        self._cpu_reader = CpuReader(self._config["cpu_readings"])
        self._cpu_utilization_dict: Dict[int, float] = {}

    def _init_frame(self):
        self._frame = build_frame(self._pm_amount, self.calc_max_snapshots())
        self._snapshots = self._frame.snapshots

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

        self._cpu_utilization_dict = self._cpu_reader.items(tick=tick)

        for vm in self._vm_item_picker.items(tick):
            vm_info = VirtualMachine(
                id=vm.vm_id,
                cpu_cores_requirement=vm.vm_cpu_cores,
                memory_requirement=vm.vm_memory,
                lifetime=vm.vm_deleted - vm.timestamp + 1
            )
            vm_info.add_utilization(cpu_utilization=self._cpu_utilization_dict[vm.vm_id])
            vm_req_payload: VmRequestPayload = VmRequestPayload(
                vm_info=vm_info,
                remaining_buffer_time=self._buffer_time_budget
            )
            vm_requirement_event = self._event_buffer.gen_cascade_event(
                tick=tick,
                event_type=Events.REQUIREMENTS,
                payload=vm_req_payload
            )
            self._event_buffer.insert_event(event=vm_requirement_event)
            self._total_vm_requests += 1

    def post_step(self, tick: int):
        # Update energy to the environment metrices.
        total_energy = 0.0
        for pm in self._machines:
            total_energy += pm.energy_consumption
        self._total_energy_consumption += total_energy

        return tick >= self._max_tick - 1

    def get_metrics(self) -> DocableDict:
        """Get current environment metrics information.

        Returns:
            DocableDict: Metrics information.
        """

        return DocableDict(
            metrics_desc,
            total_vm_requirements=self._total_vm_requests,
            total_energy_consumption=self._total_energy_consumption,
            success_allocation=self._successful_allocation,
            success_placement=self._successful_placement,
            failed_requirements=self._failed_placement,
            total_latency=self._total_latency,
            total_oversubscribtions=self._total_oversubscribtions
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
        for live_vm in self._live_vms.values():
            # NOTE:
            # We set every VM's finish event at deletion time - creation time + 1. Therefore, in the end tick,
            # the id doesn't exist in the cpu utilization dictionary. We give them a padding 0 instead.
            if self._tick - live_vm.start_tick >= live_vm.lifetime:
                live_vm.add_utilization(cpu_utilization=0.0)
            else:
                # NOTE: Some data could be lost. We use -1.0 to represent the missing data.
                if live_vm.id not in self._cpu_utilization_dict:
                    live_vm.add_utilization(cpu_utilization=-1.0)
                else:
                    live_vm.add_utilization(cpu_utilization=self._cpu_utilization_dict[live_vm.id])
                    live_vm.cpu_utilization = live_vm.get_utilization(cur_tick=self._tick)

        for pending_vm_payload in self._pending_vm_req_payload.values():
            pending_vm = pending_vm_payload.vm_info
            if pending_vm.id not in self._cpu_utilization_dict:
                pending_vm.add_utilization(cpu_utilization=-1.0)
            else:
                pending_vm.add_utilization(cpu_utilization=self._cpu_utilization_dict[pending_vm.id])

    def _update_pm_workload(self):
        """Update CPU utilization occupied by total VMs on each PM."""
        total_energy: float = 0.0
        for pm in self._machines:
            total_pm_cpu_cores_used: float = 0.0
            for vm_id in pm.live_vms:
                vm = self._live_vms[vm_id]
                total_pm_cpu_cores_used += vm.cpu_utilization * vm.cpu_cores_requirement / 100
            pm.cpu_utilization = total_pm_cpu_cores_used / pm.cpu_cores_capacity * 100

    def _cpu_utilization_to_energy_consumption(self, cpu_utilization: float) -> float:
        """Convert the CPU utilization to energy consumption.

        The formulation refers to https://dl.acm.org/doi/epdf/10.1145/1273440.1250665
        """
        cpu_utilization /= 100
        power: float = self._config.PM["P1"]["POWER_CURVE"]["CALIBRATION_PARAMETER"]
        busy_power = self._config.PM["P1"]["POWER_CURVE"]["BUSY_POWER"]
        idle_power = self._config.PM["P1"]["POWER_CURVE"]["IDLE_POWER"]
        return idle_power + (busy_power - idle_power) * (2 * cpu_utilization - pow(cpu_utilization, power))

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

    def _get_valid_pms(self, vm_cpu_cores_requirement: int) -> List[ValidPhysicalMachine]:
        """Check all valid PMs.

        Args: vm_cpu_cores_requirement (int): The vCPU cores requested by the VM.
        """
        # NOTE: Should we implement this logic inside the action scope?
        # TODO: In oversubscribable scenario, we should consider more situations, like
        #       the PM type (oversubscribable and non-oversubscribable).
        valid_pm_list = []
        for pm in self._machines:
            if (pm.cpu_cores_capacity - pm.cpu_allocation) >= vm_cpu_cores_requirement:
                valid_pm_list.append(ValidPhysicalMachine(
                    pm_id=pm.id,
                    remaining_cpu=pm.cpu_cores_capacity - pm.cpu_allocation,
                    remaining_mem=pm.memory_capacity - pm.memory_allocation
                ))

        return valid_pm_list

    def _on_vm_required(self, vm_requirement_event: CascadeEvent):
        """Callback when there is a VM requirement generated."""
        # Get VM data from payload.
        payload: VmRequestPayload = vm_requirement_event.payload

        vm_info: VirtualMachine = payload.vm_info
        remaining_buffer_time: int = payload.remaining_buffer_time
        # Store the payload inside business engine.
        self._pending_vm_req_payload[vm_info.id] = payload
        # Get valid pm list.
        valid_pm_list = self._get_valid_pms(vm_cpu_cores_requirement=vm_info.cpu_cores_requirement)

        if len(valid_pm_list) > 0:
            # Generate pending decision.
            decision_payload = DecisionPayload(
                valid_pms=valid_pm_list,
                vm_id=vm_info.id,
                vm_cpu_cores_requirement=vm_info.cpu_cores_requirement,
                vm_memory_requirement=vm_info.memory_requirement,
                remaining_buffer_time=remaining_buffer_time
            )
            pending_decision_event = self._event_buffer.gen_decision_event(
                tick=vm_requirement_event.tick, payload=decision_payload)
            vm_requirement_event.add_immediate_event(event=pending_decision_event)
        else:
            # Either postpone the requirement event or failed.
            self._postpone_vm_requirement(
                postpone_type=PostponeType.Resource,
                vm_id=vm_info.id,
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
        pm.cpu_allocation -= vm.cpu_cores_requirement
        pm.memory_allocation -= vm.memory_requirement
        # Calculate the PM's utilization.
        pm_cpu_utilization = (
            (pm.cpu_cores_capacity * pm.cpu_utilization - vm.cpu_cores_requirement * vm.cpu_utilization)
            / pm.cpu_cores_capacity
        )
        pm.cpu_utilization = pm_cpu_utilization
        pm.remove_vm(vm_id)

        # Remove dead VM.
        self._live_vms.pop(vm_id)

        # VM placement succeed.
        self._successful_placement += 1

    def _on_action_received(self, event: CascadeEvent):
        """Callback wen we get an action from agent."""
        action = None
        if event is None or event.payload is None:
            return

        cur_tick: int = event.tick

        for action in event.payload:
            vm_id: int = action.vm_id

            if vm_id not in self._pending_vm_req_payload:
                raise Exception(f"The VM id: '{vm_id}' sent by agent is invalid.")

            if type(action) == AssignAction:
                pm_id = action.pm_id
                vm: VirtualMachine = self._pending_vm_req_payload[vm_id].vm_info
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
                finished_payload: VmFinishedPayload = VmFinishedPayload(vm.id)
                finished_event = self._event_buffer.gen_atom_event(
                    tick=cur_tick + lifetime,
                    event_type=Events.FINISHED,
                    payload=finished_payload
                )
                self._event_buffer.insert_event(event=finished_event)

                # Update PM resources requested by VM.
                pm = self._machines[pm_id]
                pm.place_vm(vm.id)
                pm.cpu_allocation += vm.cpu_cores_requirement
                pm.memory_allocation += vm.memory_requirement
                # Calculate the PM's utilization.
                pm_cpu_utilization = (
                    (pm.cpu_cores_capacity * pm.cpu_utilization + vm.cpu_cores_requirement * vm.cpu_utilization)
                    / pm.cpu_cores_capacity
                )
                pm.cpu_utilization = pm_cpu_utilization
                pm.energy_consumption = self._cpu_utilization_to_energy_consumption(cpu_utilization=pm.cpu_utilization)
                self._successful_allocation += 1
            elif type(action) == PostponeAction:
                postpone_frequency = action.postpone_frequency
                remaining_buffer_time = self._pending_vm_req_payload[vm_id].remaining_buffer_time
                # Either postpone the requirement event or failed.
                self._postpone_vm_requirement(
                    postpone_type=PostponeType.Agent,
                    vm_id=vm_id,
                    remaining_buffer_time=remaining_buffer_time - postpone_frequency * self._config["delay_duration"]
                )
