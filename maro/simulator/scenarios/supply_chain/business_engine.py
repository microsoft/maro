# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from collections import defaultdict
from typing import Dict, List

from maro.backends.frame import FrameBase
from maro.event_buffer import CascadeEvent, MaroEvents
from maro.simulator.scenarios import AbsBusinessEngine

from .actions import ConsumerAction, ManufactureAction
from .objects import SupplyChainEntity
from .parser import ConfigParser, SupplyChainConfiguration
from .units import ConsumerUnit, DistributionUnit, ManufactureUnit, ProductUnit
from .world import World


class SupplyChainBusinessEngine(AbsBusinessEngine):
    def __init__(self, **kwargs) -> None:
        super().__init__(scenario_name="supply_chain", **kwargs)

        self._register_events()

        self._build_world()
        self._collect_units()

        self._product_units: List[ProductUnit] = []
        self._tick: int = 0

        # Prepare product unit for later using.
        for unit in self.world.units.values():
            if isinstance(unit, ProductUnit):
                self._product_units.append(unit)

        self._frame = self.world.frame

        self._node_mapping = self.world.get_node_mapping()

        self._metrics_cache = None

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

        self._tick = tick

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

    def _collect_units(self) -> None:
        self._consumer_dict: Dict[int, ConsumerUnit] = {}
        self._manufacture_dict: Dict[int, ManufactureUnit] = {}
        self._distribution_dict: Dict[int, DistributionUnit] = {}
        for unit in self.world.get_units_by_root_type(ConsumerUnit):
            assert isinstance(unit, ConsumerUnit)
            self._consumer_dict[unit.id] = unit
        for unit in self.world.get_units_by_root_type(ManufactureUnit):
            assert isinstance(unit, ManufactureUnit)
            self._manufacture_dict[unit.id] = unit
        for unit in self.world.get_units_by_root_type(DistributionUnit):
            assert isinstance(unit, DistributionUnit)
            self._distribution_dict[unit.id] = unit

    def _on_action_received(self, event: CascadeEvent) -> None:
        tick = event.tick

        # Handle actions
        actions = event.payload
        assert isinstance(actions, list)

        consumer_actions_by_unit: Dict[int, List[ConsumerAction]] = defaultdict(list)
        manufacture_actions_by_unit: Dict[int, List[ManufactureAction]] = defaultdict(list)
        for action in actions:
            if isinstance(action, ConsumerAction):
                consumer_actions_by_unit[action.id].append(action)
            elif isinstance(action, ManufactureAction):
                manufacture_actions_by_unit[action.id].append(action)
            else:
                raise ValueError(f"Invalid action type {type(action)}.")

        # Allocate consumer & manufacture actions
        for unit_id, consumer_actions in consumer_actions_by_unit.items():
            consumer_unit = self._consumer_dict[unit_id]
            consumer_unit.process_actions(consumer_actions, tick)
        for unit_id, manufacture_actions in manufacture_actions_by_unit.items():
            assert len(manufacture_actions) == 1  # Manufacture unit should have at most one action
            manufacture_unit = self._manufacture_dict[unit_id]
            manufacture_unit.process_action(tick, manufacture_actions[0])

        self._step_after_action_received(tick)

    def _step_after_action_received(self, tick: int) -> None:
        # Process distributions
        for distribution_unit in self._distribution_dict.values():
            distribution_unit.try_schedule_orders(tick)
            distribution_unit.handle_arrival_payloads(tick)

        # Do manufacturing
        for manufacture_unit in self._manufacture_dict.values():
            manufacture_unit.execute_manufacture(tick)

    def get_metrics(self) -> dict:
        if self._metrics_cache is None:
            self._metrics_cache = {
                "products": {
                    product.id: {
                        "sale_mean": product.get_sale_mean(),
                        "sale_std": product.get_sale_std(),
                        "demand_mean": product.get_demand_mean(),
                        "demand_std": product.get_demand_std(),
                        "selling_price": product.get_max_sale_price(),
                        "pending_order_daily":
                            product.consumer.get_pending_order_daily(self._tick)
                            if product.consumer is not None else None,
                        "waiting_order_quantity":
                            product.consumer.waiting_order_quantity if product.consumer is not None else None,
                    } for product in self._product_units
                },
                "facilities": {
                    facility.id: {
                        "in_transit_orders": facility.get_in_transit_orders(),
                        "pending_order":
                            defaultdict(int) if facility.distribution is None
                            else facility.distribution.pending_product_quantity,
                    } for facility in self.world.facilities.values()
                }
            }

        return self._metrics_cache

    def get_agent_idx_list(self) -> List[int]:
        return []

    def set_seed(self, seed: int) -> None:
        pass
