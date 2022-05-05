# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import collections
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Set, Tuple

from maro.backends.frame import FrameBase

from maro.event_buffer import CascadeEvent, MaroEvents
from maro.simulator.scenarios import AbsBusinessEngine

from .actions import SupplyChainAction
from .objects import SupplyChainEntity
from .parser import ConfigParser, SupplyChainConfiguration
from .units import ConsumerUnit, DistributionUnit, ManufactureUnit, ProductUnit
from .world import World

ACTIONS_PROCESS_DONE = "actions_process_done"
TASK_CONSUMER_ACTION_TEMPLATE = "consumer_{}_process_actions"
TASK_MANUFACTURE_ACTION_TEMPLATE = "manufacture_{}_process_action"


@dataclass
class DAGTask:
    name: str
    func: Callable
    args: tuple
    kwargs: dict

    def execute(self, tick: int) -> None:
        if self.func is not None:
            self.func(*self.args, **self.kwargs, **{"tick": tick})


class DAGTaskScheduler(object):
    def __init__(self) -> None:
        super(DAGTaskScheduler, self).__init__()

        self._task_dict: Dict[str, DAGTask] = {}  # "Vertex" in the DAG
        self._dependence: List[Tuple[str, str]] = []  # "Edge" in the DAG
        self._topological_order: List[str] = []
        self._skip_task_set: Set[str] = set([])

    def add_task(self, name: str, func: Callable = None, args: tuple = None, kwargs: dict = None) -> None:
        task = DAGTask(
            name=name,
            func=func,
            args=() if args is None else args,
            kwargs={} if kwargs is None else kwargs,
        )
        self._task_dict[name] = task

    def add_dependence(self, upstream: str, downstream: str) -> None:
        assert upstream != downstream, "Self loop is not allowed."
        self._dependence.append((upstream, downstream))

    def update_arguments(self, name: str, args: tuple = None, kwargs: dict = None) -> None:
        self._task_dict[name].args = () if args is None else args
        self._task_dict[name].kwargs = {} if kwargs is None else kwargs

    def make_topological_order(self) -> None:
        in_degree = collections.Counter()
        edge_dict = collections.defaultdict(list)
        for upstream, downstream in self._dependence:
            in_degree[downstream] += 1
            edge_dict[upstream].append(downstream)

        queue = collections.deque()
        for name, task in self._task_dict.items():
            if in_degree[name] == 0:
                queue.append(name)

        self._topological_order = []
        while queue:
            upstream = queue.popleft()
            self._topological_order.append(upstream)
            for downstream in edge_dict[upstream]:
                in_degree[downstream] -= 1
                if in_degree[downstream] == 0:
                    queue.append(downstream)

        assert len(self._topological_order) == len(self._task_dict), "There are loops in the graph. Cannot form a DAG."

    def skip_task_for_one_tick(self, name: str) -> None:
        self._skip_task_set.add(name)

    def run(self, tick: int) -> None:
        for name in self._topological_order:
            if name not in self._skip_task_set:
                self._task_dict[name].execute(tick)

        self._skip_task_set.clear()


class SupplyChainBusinessEngine(AbsBusinessEngine):
    def __init__(self, **kwargs) -> None:
        super().__init__(scenario_name="supply_chain", **kwargs)

        self._register_events()

        self._build_world()

        self._product_units: List[ProductUnit] = []

        # Prepare product unit for later using.
        for unit in self.world.units.values():
            if isinstance(unit, ProductUnit):
                self._product_units.append(unit)

        self._frame = self.world.frame

        self._node_mapping = self.world.get_node_mapping()

        self._metrics_cache = None

        self._dag_task_scheduler = DAGTaskScheduler()
        self._build_dag()

    @property
    def frame(self) -> FrameBase:
        return self._frame

    @property
    def snapshots(self) -> object:
        return self._frame.snapshots

    @property
    def configs(self) -> SupplyChainConfiguration:
        return self.world.configs

    def step(self, tick: int) -> None:
        # Clear the metrics cache.
        self._metrics_cache = None

        """
        Initialize info & status that would be used in step(), including:
        - update SKU price
        - initialize internal status
        """
        for facility in self.world.facilities.values():
            facility.pre_step(tick)

        # TODO: need to order Facility or not?
        # Call step functions by facility
        for facility in self.world.facilities.values():
            facility.step(tick)

        """
        Flush states to frame before generating decision event.
        . The processing logic requires that: DO NOT call flush_states() immediately after step().
        E.g. the ProductUnit.flush_states() should be called after the DistributionUnit.step().
        """
        for facility in self.world.facilities.values():
            facility.flush_states()

        # We do not have payload here.
        decision_event = self._event_buffer.gen_decision_event(tick, None)

        self._event_buffer.insert_event(decision_event)

    def post_step(self, tick: int) -> bool:
        # Call post_step functions by facility.
        for facility in self.world.facilities.values():
            facility.post_step(tick)

        for facility in self.world.facilities.values():
            facility.flush_states()

        self._frame.take_snapshot(self.frame_index(tick))

        return tick + 1 == self._max_tick

    def reset(self, keep_seed: bool = False) -> None:
        self._frame.reset()

        if self._frame.snapshots:
            self._frame.snapshots.reset()

        # Call reset functions by facility.
        for facility in self.world.facilities.values():
            facility.reset()

    def get_node_mapping(self) -> dict:
        return self._node_mapping

    def get_entity_list(self) -> List[SupplyChainEntity]:
        """Get a list of entities.

        Returns:
            list: List of entities.
        """
        return self.world.entity_list

    def _register_events(self) -> None:
        self._event_buffer.register_event_handler(MaroEvents.TAKE_ACTION, self._on_action_received)

    def _build_world(self) -> None:
        self.update_config_root_path(__file__)

        # Core configuration always in topologies folder.
        be_root = os.path.split(os.path.realpath(__file__))[0]
        core_config = os.path.join(be_root, "topologies", "core.yml")

        parser = ConfigParser(core_config, self._config_path)

        conf = parser.parse()

        self.world = World()

        self.world.build(conf, self.calc_max_snapshots(), self._max_tick)

    def _build_dag(self) -> None:
        """Build the task DAG that will be used in `_on_action_received`
        """

        # Get all consumer units & manufacture units of the world.
        # They will be reused in `_on_action_received`.
        self._consumer_units: List[ConsumerUnit] = []  # For reuse
        self._manufacture_units: List[ManufactureUnit] = []  # For reuse
        for unit in self.world.get_units_by_root_type(ConsumerUnit):
            assert isinstance(unit, ConsumerUnit)
            self._consumer_units.append(unit)
        for unit in self.world.get_units_by_root_type(ManufactureUnit):
            assert isinstance(unit, ManufactureUnit)
            self._manufacture_units.append(unit)

        # Dummy task. Used to complete the DAG's structure
        # Executing ACTIONS_PROCESS_DONE means all consumer actions are processed.
        self._dag_task_scheduler.add_task(ACTIONS_PROCESS_DONE)

        # Process consumer actions
        for unit in self._consumer_units:
            task_name = TASK_CONSUMER_ACTION_TEMPLATE.format(unit.id)
            self._dag_task_scheduler.add_task(task_name, unit.process_actions)
            self._dag_task_scheduler.add_dependence(task_name, ACTIONS_PROCESS_DONE)

        # Process manufacture actions.
        # All manufacture actions could be processed only after all consumer actions are processed.
        # So all manufacture actions depends on ACTIONS_PROCESS_DONE.
        for unit in self._manufacture_units:
            # Allocate manufacture actions
            task_name = TASK_MANUFACTURE_ACTION_TEMPLATE.format(unit.id)
            self._dag_task_scheduler.add_task(task_name, unit.process_action)
            # Execute manufacturing
            task_name = f"manufacture_{unit.id}_execute_manufacture"
            self._dag_task_scheduler.add_task(task_name, unit.execute_manufacture)
            self._dag_task_scheduler.add_dependence(ACTIONS_PROCESS_DONE, task_name)

        # All distribution units try to schedule orders and handle arrival payloads.
        # Order scheduling must be executed after all consumer actions are processed,
        # so let them depend on ACTIONS_PROCESS_DONE.
        # For each distribution unit, do order scheduling first, then handle arrival payloads.
        for unit in self.world.units_by_type[DistributionUnit]:
            assert isinstance(unit, DistributionUnit)
            schedule_task_name = f"distribution_{unit.id}_try_schedule_orders"
            self._dag_task_scheduler.add_task(schedule_task_name, unit.try_schedule_orders)
            arrival_task_name = f"distribution_{unit.id}_handle_arrival_payloads"
            self._dag_task_scheduler.add_task(arrival_task_name, unit.handle_arrival_payloads)

            self._dag_task_scheduler.add_dependence(ACTIONS_PROCESS_DONE, schedule_task_name)
            self._dag_task_scheduler.add_dependence(schedule_task_name, arrival_task_name)

        # Demonstration:
        #                                           Manufacture actions
        #                                         /
        # Consumer actions - ACTIONS_PROCESS_DONE - Order scheduling - handle arrival payloads

        # Generate topological order
        self._dag_task_scheduler.make_topological_order()

    def _on_action_received(self, event: CascadeEvent) -> None:
        """DAG and topological order has already been generated.

        Update all parameters of the action related tasks, and launch the workflow on DAG.
        """
        tick = event.tick

        # Handle actions
        actions = event.payload
        assert isinstance(actions, list)

        # Aggregate actions
        actions_by_unit: Dict[int, List[SupplyChainAction]] = collections.defaultdict(list)
        for i, action in enumerate(actions):
            assert isinstance(action, SupplyChainAction)
            actions_by_unit[action.id].append(action)

        # Consumer actions
        for unit in self._consumer_units:
            actions = actions_by_unit.get(unit.id, [])
            task_name = TASK_CONSUMER_ACTION_TEMPLATE.format(unit.id)
            if len(actions) == 0:
                self._dag_task_scheduler.skip_task_for_one_tick(task_name)
            else:
                self._dag_task_scheduler.update_arguments(task_name, kwargs={"actions": actions})

        # Manufacture actions
        for unit in self._manufacture_units:
            actions = actions_by_unit.get(unit.id, [])
            task_name = TASK_MANUFACTURE_ACTION_TEMPLATE.format(unit.id)
            if len(actions) == 0:
                self._dag_task_scheduler.skip_task_for_one_tick(task_name)
            elif len(actions) == 1:
                self._dag_task_scheduler.update_arguments(task_name, kwargs={"action": actions[0]})
            else:
                raise ValueError(f"Manufacture {unit.id} receives more than 1 ({len(actions)}) actions in one tick.")

        # Run all tasks
        self._dag_task_scheduler.run(tick)

    def get_metrics(self) -> dict:
        if self._metrics_cache is None:
            self._metrics_cache = {
                "products": {
                    product.id: {
                        "sale_mean": product.get_sale_mean(),
                        "sale_std": product.get_sale_std(),
                        "selling_price": product.get_max_sale_price(),
                        "pending_order_daily":
                            None if product.consumer is None else product.consumer.pending_order_daily,
                    } for product in self._product_units
                },
                "facilities": {
                    facility.id: {
                        "in_transit_orders": facility.get_in_transit_orders(),
                        "pending_order":
                            None if facility.distribution is None
                            else facility.distribution.get_pending_product_quantities(),
                    } for facility in self.world.facilities.values()
                }
            }

        return self._metrics_cache
