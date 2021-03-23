# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import shutil
import tarfile
from typing import Dict, List

from yaml import safe_load

from maro.backends.frame import FrameBase, SnapshotList
from maro.cli.data_pipeline.utils import StaticParameter, download_file
from maro.data_lib import BinaryReader
from maro.event_buffer import CascadeEvent, EventBuffer, MaroEvents
from maro.simulator.scenarios.abs_business_engine import AbsBusinessEngine
from maro.simulator.scenarios.helpers import DocableDict
from maro.utils.logger import CliLogger
from maro.utils.utils import convert_dottable

from .common import AllocateAction, DecisionPayload, Latency, PostponeAction, VmRequestPayload
from .cpu_reader import CpuReader
from .enums import Events, PmState, PostponeType, VmCategory
from .frame_builder import build_frame
from .physical_machine import PhysicalMachine
from .virtual_machine import VirtualMachine

metrics_desc = """
VM scheduling metrics used provide statistics information until now.
It contains following keys:

total_vm_requests (int): Total VM requests.
total_incomes (float): Accumulative total incomes at current tick (unit: dollar).
                        (if there is overload phenomenon happened, return the money for each failed VM).
energy_consumption_cost (float): Accumulative energy consumption cost at current tick (unit: dollar).
total_profit (float): Accumulative total profit at current tick (total_incomes-energy_consumption_cost) (unit: dollar).
total_energy_consumption (float): Accumulative total PM energy consumption (unit: KWh).
successful_allocation (int): Accumulative successful VM allocation until now.
successful_completion (int): Accumulative successful completion of tasks.
failed_allocation (int): Accumulative failed VM allocation until now.
failed_completion (int): Accumulative failed VM completion due to PM overloading.
total_latency (Latency): Accumulative used buffer time until now.
total_oversubscriptions (int): Accumulative over-subscriptions. The unit is PM amount * tick.
total_overload_pms (int): Accumulative overload pms. The unit is PM amount * tick.
total_overload_vms (int): Accumulative VMs on overload pms. The unit is VM amount * tick.
"""

logger = CliLogger(name=__name__)


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

        # Initialize environment metrics.
        self._init_metrics()
        # Load configurations.
        self._load_configs()
        self._register_events()

        self._init_frame()
        # Initialize simulation data.
        self._init_data()

        # Data center structure for quick accessing.
        self._init_structure()

        # All living VMs.
        self._live_vms: Dict[int, VirtualMachine] = {}
        # All request payload of the pending decision VMs.
        # NOTE: Need naming suggestestion.
        self._pending_vm_request_payload: Dict[int, VmRequestPayload] = {}

        self._vm_reader = BinaryReader(self._config.VM_TABLE)
        self._vm_item_picker = self._vm_reader.items_tick_picker(self._start_tick, self._max_tick, time_unit="s")

        self._cpu_reader = CpuReader(data_path=self._config.CPU_READINGS, start_tick=self._start_tick)

        self._tick: int = 0
        self._pending_action_vm_id: int = 0

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
            self._config = convert_dottable(safe_load(fp))

        self._delay_duration: int = self._config.DELAY_DURATION
        self._buffer_time_budget: int = self._config.BUFFER_TIME_BUDGET
        self._ticks_per_hour: float = self._config.TICKS_PER_HOUR
        # Oversubscription rate.
        self._max_cpu_oversubscription_rate: float = self._config.MAX_CPU_OVERSUBSCRIPTION_RATE
        self._max_memory_oversubscription_rate: float = self._config.MAX_MEM_OVERSUBSCRIPTION_RATE
        self._max_utilization_rate: float = self._config.MAX_UTILIZATION_RATE

        # Price parameters
        self._price_per_cpu_cores_per_hour: float = self._config.PRICE_PER_CPU_CORES_PER_HOUR
        self._price_per_memory_per_hour: float = self._config.PRICE_PER_MEMORY_PER_HOUR
        self._unit_energy_price_per_kwh: float = self._config.UNIT_ENERGY_PRICE_PER_KWH
        self._power_usage_efficiency: float = self._config.POWER_USAGE_EFFICIENCY

        # Load PM related configs.
        self._region_amount: int = sum(
            len(item) for item in self._find_item(key="region", dictionary=self._config.architecture)
        )
        self._zone_amount: int = sum(
            len(item) for item in self._find_item(key="zone", dictionary=self._config.architecture)
        )
        self._data_center_amount: int = sum(
            len(item) for item in self._find_item(key="data_center", dictionary=self._config.architecture)
        )
        # Cluster amount dict.
        cluster_amount_dict = {}
        for cluster_list in self._find_item(key="cluster", dictionary=self._config.architecture):
            for cluster in cluster_list:
                cluster_amount_dict[cluster['type']] = (
                    cluster_amount_dict.get(cluster['type'], 0) + cluster['cluster_amount']
                )
        # Summation of cluster amount.
        self._cluster_amount: int = sum(value for value in cluster_amount_dict.values())

        # Rack amount dict.
        rack_amount_dict = {}
        for cluster_list in self._find_item(key="cluster", dictionary=self._config.components):
            for cluster in cluster_list:
                for rack in cluster['rack']:
                    rack_amount_dict[rack['rack_type']] = (
                        rack_amount_dict.get(rack['rack_type'], 0)
                        + cluster_amount_dict[cluster['type']] * rack['rack_amount']
                    )
        # Summation of rack amount.
        self._rack_amount: int = sum(value for value in rack_amount_dict.values())

        # PM amount dict.
        pm_amount_dict = {}
        for rack in self._config.components.rack:
            for pm in rack['pm']:
                pm_amount_dict[pm['pm_type']] = (
                    pm_amount_dict.get(pm['pm_type'], 0)
                    + rack_amount_dict[rack['type']] * pm['pm_amount']
                )
        # Summation of pm amount.
        self._pm_amount: int = sum(value for value in pm_amount_dict.values())

        self._kill_all_vms_if_overload: bool = self._config.KILL_ALL_VMS_IF_OVERLOAD

    def _init_metrics(self):
        # Env metrics.
        self._total_vm_requests: int = 0
        self._total_incomes: float = 0.0
        self._total_profit: float = 0.0
        self._energy_consumption_cost: float = 0.0
        self._total_energy_consumption: float = 0.0
        self._successful_allocation: int = 0
        self._successful_completion: int = 0
        self._failed_allocation: int = 0
        self._failed_completion: int = 0
        self._total_latency: Latency = Latency()
        self._total_oversubscriptions: int = 0
        self._total_overload_pms: int = 0
        self._total_overload_vms: int = 0

    def _init_data(self):
        """If the file does not exist, then trigger the short data pipeline to download the processed data."""
        vm_table_data_path = self._config.VM_TABLE
        if vm_table_data_path.startswith("~"):
            vm_table_data_path = os.path.expanduser(vm_table_data_path)

        cpu_readings_data_path = self._config.CPU_READINGS
        if cpu_readings_data_path.startswith("~"):
            cpu_readings_data_path = os.path.expanduser(cpu_readings_data_path)

        # Testing file.
        if vm_table_data_path.startswith("tests") or cpu_readings_data_path.startswith("tests"):
            pass
        elif (not os.path.exists(vm_table_data_path)) or (not os.path.exists(cpu_readings_data_path)):
            logger.info_green("Lack data. Start preparing data.")
            self._download_processed_data()
            logger.info_green("Data preparation is finished.")

    def _find_item(self, key: str, dictionary: dict) -> int:
        for k, v in dictionary.items():
            if k == key:
                yield v
            elif isinstance(v, list):
                for item in v:
                    for result in self._find_item(key, item):
                        yield result
            elif isinstance(v, dict):
                for result in self._find_item(key, v):
                    yield result

    def _init_structure(self):
        """Initialize the whole pm structure."""
        # TODO: Improve the scalability. Like the use of multiple PM sets.
        self._regions = self._frame.regions
        self._zones = self._frame.zones
        self._data_centers = self._frame.data_centers
        self._clusters = self._frame.clusters
        self._racks = self._frame.racks
        self._machines = self._frame.pms
        # PM type dictionary.
        self._cluster_config_dict = {
            cluster['type']: {
                rack['rack_type']: rack['rack_amount'] for rack in cluster['rack']
            }
            for cluster in self._config.components.cluster
        }
        self._rack_config_dict = {
            rack['type']: {
                pm['pm_type']: pm['pm_amount'] for pm in rack['pm']
            }
            for rack in self._config.components.rack
        }
        self._pm_config_dict: dict = {
            pm_type: pm_dict for pm_type, pm_dict in enumerate(self._config.components.pm)
        }
        # Ids.
        self._region_id = 0
        self._zone_id = 0
        self._data_center_id = 0
        self._cluster_id = 0
        self._rack_id = 0
        self._pm_id = 0
        # Initialize regions.
        self._init_regions()

    def _init_regions(self):
        """Initialize the regions based on the config setting. The regions id starts from 0."""
        for region_list in self._find_item("region", self._config.architecture):
            for region_dict in region_list:
                # Initialize zones.
                start_zone_id = self._init_zones(
                    zone_list=region_dict["zone"]
                )
                region = self._regions[self._region_id]
                region.name = region_dict["name"]
                region.zone_list = [id for id in range(start_zone_id, self._zone_id)]
                total_machine_num = sum(
                    self._zones[id].total_machine_num for id in region.zone_list
                )
                region.set_init_state(
                    id=self._region_id,
                    total_machine_num=total_machine_num
                )
                self._region_id += 1

    def _init_zones(self, zone_list: list):
        """Initialize the zones based on the config setting. The zone id starts from 0."""
        start_zone_id = self._zone_id
        for zone_dict in zone_list:
            # Initialize data centers.
            start_data_center_id = self._init_data_centers(
                data_center_list=zone_dict["data_center"]
            )
            zone = self._zones[self._zone_id]
            zone.name = zone_dict["name"]
            zone.data_center_list = [id for id in range(start_data_center_id, self._data_center_id)]
            total_machine_num = sum(
                self._data_centers[id].total_machine_num for id in zone.data_center_list
            )
            zone.set_init_state(
                id=self._zone_id,
                region_id=self._region_id,
                total_machine_num=total_machine_num
            )

            self._zone_id += 1

        return start_zone_id

    def _init_data_centers(self, data_center_list: list):
        """Initialize the data_centers based on the config setting. The data_center id starts from 0."""
        start_data_center_id = self._data_center_id
        for data_center_dict in data_center_list:
            # Initialize clusters.
            start_cluster_id = self._init_clusters(
                cluster_list=data_center_dict["cluster"]
            )
            data_center = self._data_centers[self._data_center_id]
            data_center.name = data_center_dict["name"]
            data_center.cluster_list = [id for id in range(start_cluster_id, self._cluster_id)]
            total_machine_num = sum(
                self._clusters[id].total_machine_num for id in data_center.cluster_list
            )
            data_center.set_init_state(
                id=self._data_center_id,
                region_id=self._region_id,
                zone_id=self._zone_id,
                total_machine_num=total_machine_num
            )
            self._data_center_id += 1

        return start_data_center_id

    def _init_clusters(self, cluster_list: list):
        """Initialize the clusters based on the config setting. The cluster id starts from 0."""
        start_cluster_id = self._cluster_id
        for cluster in cluster_list:
            cluster_type = cluster['type']
            cluster_amount = cluster['cluster_amount']
            while cluster_amount > 0:
                # Init racks.
                start_rack_id = self._init_racks(
                    rack_amount_dict=self._cluster_config_dict[cluster_type]
                )
                cluster = self._clusters[self._cluster_id]
                cluster.cluster_type = cluster_type
                cluster.rack_list = [id for id in range(start_rack_id, self._rack_id)]
                total_machine_num = sum(
                    self._racks[id].total_machine_num for id in cluster.rack_list
                )
                cluster.set_init_state(
                    id=self._cluster_id,
                    region_id=self._region_id,
                    zone_id=self._zone_id,
                    data_center_id=self._data_center_id,
                    total_machine_num=total_machine_num
                )

                cluster_amount -= 1
                self._cluster_id += 1

        return start_cluster_id

    def _init_racks(self, rack_amount_dict: dict):
        """Initialize the racks based on the config setting. The rack id starts from 0."""
        start_rack_id = self._rack_id
        for rack_type, rack_amount in rack_amount_dict.items():
            while rack_amount > 0:
                # Initialize pms.
                start_pm_id = self._init_pms(
                    pm_dict=self._rack_config_dict[rack_type]
                )
                rack = self._racks[self._rack_id]
                rack.type = rack_type
                rack.pm_list = [id for id in range(start_pm_id, self._pm_id)]
                total_machine_num = len(rack.pm_list)
                rack.set_init_state(
                    id=self._rack_id,
                    region_id=self._region_id,
                    zone_id=self._zone_id,
                    data_center_id=self._data_center_id,
                    cluster_id=self._cluster_id,
                    total_machine_num=total_machine_num
                )
                rack_amount -= 1
                self._rack_id += 1

        return start_rack_id

    def _init_pms(self, pm_dict: dict):
        """Initialize the pms based on the config setting. The pm id starts from 0."""
        start_pm_id = self._pm_id
        for pm_type, pm_amount in pm_dict.items():
            while pm_amount > 0:
                pm = self._machines[self._pm_id]
                pm.set_init_state(
                    id=self._pm_id,
                    cpu_cores_capacity=self._pm_config_dict[pm_type]["cpu"],
                    memory_capacity=self._pm_config_dict[pm_type]["memory"],
                    pm_type=pm_type,
                    region_id=self._region_id,
                    zone_id=self._zone_id,
                    data_center_id=self._data_center_id,
                    cluster_id=self._cluster_id,
                    rack_id=self._rack_id,
                    oversubscribable=PmState.EMPTY,
                    idle_energy_consumption=self._cpu_utilization_to_energy_consumption(
                        pm_type=self._pm_config_dict[pm_type],
                        cpu_utilization=0
                    )
                )

                pm_amount -= 1
                self._pm_id += 1

        return start_pm_id

    def reset(self):
        """Reset internal states for episode."""
        self._init_metrics()

        self._frame.reset()
        self._snapshots.reset()

        for pm in self._machines:
            pm.reset()

        for rack in self._racks:
            rack.reset()

        for cluster in self._clusters:
            cluster.reset()

        for data_center in self._data_centers:
            data_center.reset()

        for zone in self._zones:
            zone.reset()

        for region in self._regions:
            region.reset()

        self._live_vms.clear()
        self._pending_vm_request_payload.clear()

        self._vm_reader.reset()
        self._vm_item_picker = self._vm_reader.items_tick_picker(self._start_tick, self._max_tick, time_unit="s")

        self._cpu_reader.reset()

    def _init_frame(self):
        self._frame = build_frame(
            snapshots_num=self.calc_max_snapshots(),
            region_amount=self._region_amount,
            zone_amount=self._zone_amount,
            data_center_amount=self._data_center_amount,
            cluster_amount=self._cluster_amount,
            rack_amount=self._rack_amount,
            pm_amount=self._pm_amount
        )
        self._snapshots = self._frame.snapshots

    def step(self, tick: int):
        """Push business to next step.

        Args:
            tick (int): Current tick to process.
        """
        self._tick = tick
        # All vm's cpu utilization at current tick.
        cur_tick_cpu_utilization = self._cpu_reader.items(tick=tick)

        # Process finished VMs.
        self._process_finished_vm()
        # Update all live VMs CPU utilization.
        self._update_vm_workload(cur_tick_cpu_utilization=cur_tick_cpu_utilization)
        # Update all PM CPU utilization.
        self._update_pm_workload()
        self._update_upper_level_metrics()

        for vm in self._vm_item_picker.items(tick):
            # TODO: Batch request support.
            unit_price = self._get_unit_price(vm.vm_cpu_cores, vm.vm_memory)

            vm_info = VirtualMachine(
                id=vm.vm_id,
                cpu_cores_requirement=vm.vm_cpu_cores,
                memory_requirement=vm.vm_memory,
                lifetime=vm.vm_lifetime,
                sub_id=vm.sub_id,
                deployment_id=vm.deploy_id,
                category=VmCategory(vm.vm_category),
                unit_price=unit_price
            )

            if vm.vm_id not in cur_tick_cpu_utilization:
                raise Exception(f"The VM id: '{vm.vm_id}' does not exist at this tick.")

            vm_info.add_utilization(cpu_utilization=cur_tick_cpu_utilization[vm.vm_id])
            vm_req_payload: VmRequestPayload = VmRequestPayload(
                vm_info=vm_info,
                remaining_buffer_time=self._buffer_time_budget
            )
            vm_request_event = self._event_buffer.gen_cascade_event(
                tick=tick,
                event_type=Events.REQUEST,
                payload=vm_req_payload
            )
            self._event_buffer.insert_event(event=vm_request_event)
            self._total_vm_requests += 1

    def post_step(self, tick: int):
        # Update energy to the environment metrices.
        total_energy: float = 0.0
        total_energy_cost: float = 0.0
        for pm in self._machines:
            if pm.oversubscribable and pm.cpu_cores_allocated > pm.cpu_cores_capacity:
                self._total_oversubscriptions += 1
            total_energy += pm.energy_consumption
            # Update the energy consumption cost.
            pm_cost = (
                pm.energy_consumption
                * self._unit_energy_price_per_kwh
                * self._power_usage_efficiency
            )
            total_energy_cost += pm_cost
            # Overload PMs.
            if pm.cpu_utilization > 100:
                self._overload(pm.id, tick)
        self._total_energy_consumption += total_energy
        self._energy_consumption_cost += total_energy_cost

        # Update the incomes.
        self._update_incomes()
        # Update the profit
        self._update_profit()

        if (tick + 1) % self._snapshot_resolution == 0:
            # NOTE: We should use frame_index method to get correct index in snapshot list.
            self._frame.take_snapshot(self.frame_index(tick))

        # Stop current episode if we reach max tick.
        return tick + 1 >= self._max_tick

    def get_event_payload_detail(self) -> dict:
        """dict: Event payload details of current scenario."""
        return {
            Events.REQUEST.name: VmRequestPayload.summary_key,
            MaroEvents.PENDING_DECISION.name: DecisionPayload.summary_key
        }

    def get_agent_idx_list(self) -> List[int]:
        """Get a list of agent index."""
        pass

    def get_node_mapping(self) -> dict:
        """dict: Node mapping."""
        node_mapping = {}

        return node_mapping

    def get_vm_cpu_utilization_series(self, vm_id: int) -> List[float]:
        """Get the CPU utilization series of the specific VM by the given ID."""
        if vm_id in self._live_vms:
            return self._live_vms[vm_id].get_historical_utilization_series(cur_tick=self._tick)

        return []

    def get_metrics(self) -> DocableDict:
        """Get current environment metrics information.

        Returns:
            DocableDict: Metrics information.
        """

        return DocableDict(
            metrics_desc,
            total_vm_requests=self._total_vm_requests,
            total_incomes=self._total_incomes,
            energy_consumption_cost=self._energy_consumption_cost,
            total_profit=self._total_profit,
            total_energy_consumption=self._total_energy_consumption,
            successful_allocation=self._successful_allocation,
            successful_completion=self._successful_completion,
            failed_allocation=self._failed_allocation,
            failed_completion=self._failed_completion,
            total_latency=self._total_latency,
            total_oversubscriptions=self._total_oversubscriptions,
            total_overload_pms=self._total_overload_pms,
            total_overload_vms=self._total_overload_vms
        )

    def _register_events(self):
        # Register our own events and their callback handlers.
        self._event_buffer.register_event_handler(event_type=Events.REQUEST, handler=self._on_vm_required)
        # Generate decision event.
        self._event_buffer.register_event_handler(event_type=MaroEvents.TAKE_ACTION, handler=self._on_action_received)

    def _update_vm_workload(self, cur_tick_cpu_utilization: dict):
        """Update all live VMs CPU utilization.

        The length of VMs utilization series could be difference among all VMs,
        because index 0 represents the VM's CPU utilization at the tick it starts.
        """
        for live_vm in self._live_vms.values():
            # NOTE: Some data could be lost. We use -1.0 to represent the missing data.
            if live_vm.id not in cur_tick_cpu_utilization:
                live_vm.add_utilization(cpu_utilization=-1.0)
            else:
                live_vm.add_utilization(cpu_utilization=cur_tick_cpu_utilization[live_vm.id])
                live_vm.cpu_utilization = live_vm.get_utilization(cur_tick=self._tick)

        for pending_vm_payload in self._pending_vm_request_payload.values():
            pending_vm = pending_vm_payload.vm_info
            if pending_vm.id not in cur_tick_cpu_utilization:
                pending_vm.add_utilization(cpu_utilization=-1.0)
            else:
                pending_vm.add_utilization(cpu_utilization=cur_tick_cpu_utilization[pending_vm.id])

    def _update_upper_level_metrics(self):
        self._update_rack_metrics()
        self._update_cluster_metrics()
        self._update_data_center_metrics()
        self._update_zone_metrics()
        self._update_region_metrics()

    def _update_region_metrics(self):
        for region in self._regions:
            region.empty_machine_num = sum(
                self._zones[zone_id].empty_machine_num for zone_id in region.zone_list
            )

    def _update_zone_metrics(self):
        for zone in self._zones:
            zone.empty_machine_num = sum(
                self._data_centers[data_center_id].empty_machine_num for data_center_id in zone.data_center_list
            )

    def _update_data_center_metrics(self):
        for data_center in self._data_centers:
            data_center.empty_machine_num = sum(
                self._clusters[cluster_id].empty_machine_num for cluster_id in data_center.cluster_list
            )

    def _update_cluster_metrics(self):
        for cluster in self._clusters:
            cluster.empty_machine_num = sum(
                self._racks[rack_id].empty_machine_num for rack_id in cluster.rack_list
            )

    def _update_rack_metrics(self):
        for rack in self._racks:
            count_empty_machine: int = 0
            for pm_id in rack.pm_list:
                pm = self._machines[pm_id]
                if pm.cpu_cores_allocated == 0:
                    count_empty_machine += 1
            rack.empty_machine_num = count_empty_machine

    def _update_pm_workload(self):
        """Update CPU utilization occupied by total VMs on each PM."""
        for pm in self._machines:
            total_pm_cpu_cores_used: float = 0.0
            for vm_id in pm.live_vms:
                vm = self._live_vms[vm_id]
                total_pm_cpu_cores_used += vm.cpu_utilization * vm.cpu_cores_requirement
            pm.update_cpu_utilization(vm=None, cpu_utilization=total_pm_cpu_cores_used / pm.cpu_cores_capacity)
            pm.energy_consumption = self._cpu_utilization_to_energy_consumption(
                pm_type=self._pm_config_dict[pm.pm_type],
                cpu_utilization=pm.cpu_utilization
            )

    def _overload(self, pm_id: int, tick: int):
        """Overload logic.

        Currently only support killing all VMs on the overload PM and note them as failed allocations.
        """
        # TODO: Future features of overload modeling.
        #       1. Performance degradation
        #       2. Quiesce specific VMs.
        pm: PhysicalMachine = self._machines[pm_id]
        vm_ids: List[int] = [vm_id for vm_id in pm.live_vms]

        if self._kill_all_vms_if_overload:
            for vm_id in vm_ids:
                self._total_incomes -= self._live_vms[vm_id].get_income_till_now(tick)
                self._live_vms.pop(vm_id)

            pm.deallocate_vms(vm_ids=vm_ids)
            self._failed_completion += len(vm_ids)

        self._total_overload_vms += len(vm_ids)

    def _cpu_utilization_to_energy_consumption(self, pm_type: dict, cpu_utilization: float) -> float:
        """Convert the CPU utilization to energy consumption.

        The formulation refers to https://dl.acm.org/doi/epdf/10.1145/1273440.1250665
        """
        power: float = pm_type["power_curve"]["calibration_parameter"]
        busy_power: int = pm_type["power_curve"]["busy_power"]
        idle_power: int = pm_type["power_curve"]["idle_power"]

        cpu_utilization /= 100
        cpu_utilization = min(1, cpu_utilization)
        energy_consumption_per_hour = (
            idle_power + (busy_power - idle_power) * (2 * cpu_utilization - pow(cpu_utilization, power))
        )

        return (energy_consumption_per_hour / self._ticks_per_hour) / 1000

    def _postpone_vm_request(self, postpone_type: PostponeType, vm_id: int, remaining_buffer_time: int):
        """Postpone VM request."""
        if remaining_buffer_time >= self._delay_duration:
            if postpone_type == PostponeType.Resource:
                self._total_latency.due_to_resource += self._delay_duration
            elif postpone_type == PostponeType.Agent:
                self._total_latency.due_to_agent += self._delay_duration

            postpone_payload = self._pending_vm_request_payload[vm_id]
            postpone_payload.remaining_buffer_time -= self._delay_duration
            postpone_event = self._event_buffer.gen_cascade_event(
                tick=self._tick + self._delay_duration,
                event_type=Events.REQUEST,
                payload=postpone_payload
            )
            self._event_buffer.insert_event(event=postpone_event)
        else:
            # Fail
            # Pop out VM request payload.
            self._pending_vm_request_payload.pop(vm_id)
            # Add failed allocation.
            self._failed_allocation += 1

    def _get_valid_pms(
        self, vm_cpu_cores_requirement: int, vm_memory_requirement: int, vm_category: VmCategory
    ) -> List[int]:
        """Check all valid PMs.

        Args:
            vm_cpu_cores_requirement (int): The CPU cores requested by the VM.
            vm_memory_requirement (int): The memory requested by the VM.
            vm_category (VmCategory): The VM category. Delay-insensitive: 0, Interactive: 1, Unknown: 2.
        """
        # NOTE: Should we implement this logic inside the action scope?
        valid_pm_list = []

        # Delay-insensitive: 0, Interactive: 1, and Unknown: 2.
        if vm_category == VmCategory.INTERACTIVE or vm_category == VmCategory.UNKNOWN:
            valid_pm_list = self._get_valid_non_oversubscribable_pms(
                vm_cpu_cores_requirement=vm_cpu_cores_requirement,
                vm_memory_requirement=vm_memory_requirement
            )
        else:
            valid_pm_list = self._get_valid_oversubscribable_pms(
                vm_cpu_cores_requirement=vm_cpu_cores_requirement,
                vm_memory_requirement=vm_memory_requirement
            )

        return valid_pm_list

    def _get_valid_non_oversubscribable_pms(self, vm_cpu_cores_requirement: int, vm_memory_requirement: int) -> list:
        valid_pm_list = []
        for pm in self._machines:
            if pm.oversubscribable == PmState.EMPTY or pm.oversubscribable == PmState.NON_OVERSUBSCRIBABLE:
                # In the condition of non-oversubscription, the valid PMs mean:
                # PM allocated resource + VM allocated resource <= PM capacity.
                if (pm.cpu_cores_allocated + vm_cpu_cores_requirement <= pm.cpu_cores_capacity
                        and pm.memory_allocated + vm_memory_requirement <= pm.memory_capacity):
                    valid_pm_list.append(pm.id)

        return valid_pm_list

    def _get_valid_oversubscribable_pms(self, vm_cpu_cores_requirement: int, vm_memory_requirement: int) -> List[int]:
        valid_pm_list = []
        for pm in self._machines:
            if pm.oversubscribable == PmState.EMPTY or pm.oversubscribable == PmState.OVERSUBSCRIBABLE:
                # In the condition of oversubscription, the valid PMs mean:
                # 1. PM allocated resource + VM allocated resource <= Max oversubscription rate * PM capacity.
                # 2. PM CPU usage + VM requirements <= Max utilization rate * PM capacity.
                if (
                    (
                        pm.cpu_cores_allocated + vm_cpu_cores_requirement
                        <= self._max_cpu_oversubscription_rate * pm.cpu_cores_capacity
                    ) and (
                        pm.memory_allocated + vm_memory_requirement
                        <= self._max_memory_oversubscription_rate * pm.memory_capacity
                    ) and (
                        pm.cpu_utilization / 100 * pm.cpu_cores_capacity + vm_cpu_cores_requirement
                        <= self._max_utilization_rate * pm.cpu_cores_capacity
                    )
                ):
                    valid_pm_list.append(pm.id)

        return valid_pm_list

    def _process_finished_vm(self):
        """Release PM resource from the finished VM."""
        # Get the VM info.
        vm_id_list = []
        for vm in self._live_vms.values():
            if vm.deletion_tick == self._tick:
                # Release PM resources.
                pm: PhysicalMachine = self._machines[vm.pm_id]
                pm.cpu_cores_allocated -= vm.cpu_cores_requirement
                pm.memory_allocated -= vm.memory_requirement
                pm.deallocate_vms(vm_ids=[vm.id])
                # If the VM list is empty, switch the state to empty.
                if not pm.live_vms:
                    pm.oversubscribable = PmState.EMPTY

                vm_id_list.append(vm.id)
                # VM completed task succeed.
                self._successful_completion += 1

        # Remove dead VM.
        for vm_id in vm_id_list:
            self._live_vms.pop(vm_id)

    def _on_vm_required(self, vm_request_event: CascadeEvent):
        """Callback when there is a VM request generated."""
        # Get VM data from payload.
        payload: VmRequestPayload = vm_request_event.payload

        vm_info: VirtualMachine = payload.vm_info
        remaining_buffer_time: int = payload.remaining_buffer_time
        # Store the payload inside business engine.
        self._pending_vm_request_payload[vm_info.id] = payload
        # Get valid pm list.
        valid_pm_list = self._get_valid_pms(
            vm_cpu_cores_requirement=vm_info.cpu_cores_requirement,
            vm_memory_requirement=vm_info.memory_requirement,
            vm_category=vm_info.category
        )

        if len(valid_pm_list) > 0:
            # Generate pending decision.
            decision_payload = DecisionPayload(
                frame_index=self.frame_index(tick=self._tick),
                valid_pms=valid_pm_list,
                vm_id=vm_info.id,
                vm_cpu_cores_requirement=vm_info.cpu_cores_requirement,
                vm_memory_requirement=vm_info.memory_requirement,
                vm_sub_id=vm_info.sub_id,
                vm_category=vm_info.category,
                remaining_buffer_time=remaining_buffer_time
            )
            self._pending_action_vm_id = vm_info.id
            pending_decision_event = self._event_buffer.gen_decision_event(
                tick=vm_request_event.tick, payload=decision_payload)
            vm_request_event.add_immediate_event(event=pending_decision_event)
        else:
            # Either postpone the requirement event or failed.
            self._postpone_vm_request(
                postpone_type=PostponeType.Resource,
                vm_id=vm_info.id,
                remaining_buffer_time=remaining_buffer_time
            )

    def _on_action_received(self, event: CascadeEvent):
        """Callback wen we get an action from agent."""
        action = None
        if event is None or event.payload is None:
            self._pending_vm_request_payload.pop(self._pending_action_vm_id)
            return

        cur_tick: int = event.tick

        for action in event.payload:
            vm_id: int = action.vm_id

            if vm_id not in self._pending_vm_request_payload:
                raise Exception(f"The VM id: '{vm_id}' sent by agent is invalid.")

            if type(action) == AllocateAction:
                pm_id = action.pm_id
                vm: VirtualMachine = self._pending_vm_request_payload[vm_id].vm_info
                lifetime = vm.lifetime

                # Update VM information.
                vm.pm_id = pm_id
                vm.creation_tick = cur_tick
                vm.deletion_tick = cur_tick + lifetime
                vm.cpu_utilization = vm.get_utilization(cur_tick=cur_tick)

                # Pop out the VM from pending requests and add to live VM dict.
                self._pending_vm_request_payload.pop(vm_id)
                self._live_vms[vm_id] = vm

                # Update PM resources requested by VM.
                pm = self._machines[pm_id]

                # Empty pm (init state).
                if pm.oversubscribable == PmState.EMPTY:
                    # Delay-Insensitive: oversubscribable.
                    if vm.category == VmCategory.DELAY_INSENSITIVE:
                        pm.oversubscribable = PmState.OVERSUBSCRIBABLE
                    # Interactive or Unknown: non-oversubscribable
                    else:
                        pm.oversubscribable = PmState.NON_OVERSUBSCRIBABLE

                pm.allocate_vms(vm_ids=[vm.id])
                pm.cpu_cores_allocated += vm.cpu_cores_requirement
                pm.memory_allocated += vm.memory_requirement
                pm.update_cpu_utilization(
                    vm=vm,
                    cpu_utilization=None
                )
                pm.energy_consumption = self._cpu_utilization_to_energy_consumption(
                    pm_type=self._pm_config_dict[pm.pm_type],
                    cpu_utilization=pm.cpu_utilization
                )
                self._successful_allocation += 1

            elif type(action) == PostponeAction:
                postpone_step = action.postpone_step
                remaining_buffer_time = self._pending_vm_request_payload[vm_id].remaining_buffer_time
                # Either postpone the requirement event or failed.
                self._postpone_vm_request(
                    postpone_type=PostponeType.Agent,
                    vm_id=vm_id,
                    remaining_buffer_time=remaining_buffer_time - postpone_step * self._delay_duration
                )

    def _update_incomes(self):
        for vm in self._live_vms.values():
            self._total_incomes += vm.unit_price

    def _update_profit(self):
        self._total_profit = self._total_incomes - self._energy_consumption_cost

    def _get_unit_price(self, cpu_cores, memory):
        return (
            self._price_per_cpu_cores_per_hour * cpu_cores
            + self._price_per_memory_per_hour * memory
        ) / self._ticks_per_hour

    def _download_processed_data(self):
        """Build processed data."""
        data_root = StaticParameter.data_root
        build_folder = os.path.join(data_root, self._scenario_name, ".build", self._topology)

        source = self._config.PROCESSED_DATA_URL
        download_file_name = source.split('/')[-1]
        download_file_path = os.path.join(build_folder, download_file_name)

        # Download file from the Azure blob storage.
        if not os.path.exists(download_file_path):
            logger.info_green(f"Downloading data from {source} to {download_file_path}.")
            download_file(source=source, destination=download_file_path)
        else:
            logger.info_green("File already exists, skipping download.")

        # Unzip files.
        logger.info_green(f"Unzip {download_file_path} to {build_folder}")
        tar = tarfile.open(download_file_path, "r:gz")
        tar.extractall(path=build_folder)
        tar.close()

        # Move to the correct path.
        for _, directories, _ in os.walk(build_folder):
            for directory in directories:
                unzip_file = os.path.join(build_folder, directory)
                logger.info_green(f"Move files to {build_folder} from {unzip_file}")
                for file_name in os.listdir(unzip_file):
                    if file_name.endswith(".bin"):
                        shutil.move(os.path.join(unzip_file, file_name), build_folder)

        os.rmdir(unzip_file)
