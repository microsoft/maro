# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod
import math, random
from typing import List, Optional

from examples.supply_chain.rl.config import OR_NUM_CONSUMER_ACTIONS, workflow_settings
from maro.rl.policy import RuleBasedPolicy


VLT_BUFFER_DAYS = workflow_settings["or_policy_vlt_buffer_days"]


class DummyPolicy(RuleBasedPolicy):
    def _rule(self, states: List[dict]) -> list:
        return [None] * len(states)


class ManufacturerBaselinePolicy(RuleBasedPolicy):
    def _rule(self, states: List[dict]) -> List[int]:
        return [500] * len(states)


class ConsumerBasePolicy(RuleBasedPolicy):
    def __init__(self, name: str) -> None:
        super().__init__(name)

        self._replenishment_threshold: Optional[float] = None

    def _take_action_mask(self, state: dict) -> bool:
        self._booked_quantity = state["product_level"] + state["in_transition_quantity"] - state["to_distributed_orders"]
        storage_booked_quantity = state["storage_utilization"] + state["storage_in_transition_quantity"]

        expected_vlt = round(VLT_BUFFER_DAYS * state["max_vlt"], 0)
        self._replenishment_threshold = (
            expected_vlt * state["sale_mean"]
            + math.sqrt(expected_vlt) * state["sale_std"] * state["service_level_ppf"]
        )

        capacity_mask = storage_booked_quantity <= state["storage_capacity"]
        replenishment_mask = self._booked_quantity <= self._replenishment_threshold

        return capacity_mask and replenishment_mask

    @abstractmethod
    def _get_action_quantity(self, state: dict) -> int:
        raise NotImplementedError

    def _rule(self, states: List[dict]) -> List[int]:
        return [
            self._take_action_mask(state) * self._get_action_quantity(state)
            for state in states
        ]


class ConsumerBaselinePolicy(ConsumerBasePolicy):
    def _get_action_quantity(self, state: dict) -> int:
        return random.randint(0, OR_NUM_CONSUMER_ACTIONS - 1)


class ConsumerEOQPolicy(ConsumerBasePolicy):
    def _get_action_quantity(self, state: dict) -> int:
        quantity = math.sqrt(2 * state["sale_mean"] * state["unit_order_cost"] / state["unit_storage_cost"])
        quantity /= (state["sale_mean"] + 1e-8)
        return min(int(quantity), OR_NUM_CONSUMER_ACTIONS - 1)


class ConsumerMinMaxPolicy(ConsumerBasePolicy):
    def _get_action_quantity(self, state: dict) -> int:
        quantity = (self._replenishment_threshold - self._booked_quantity)
        quantity /= (state["sale_mean"] + 1e-8)
        return min(int(quantity), OR_NUM_CONSUMER_ACTIONS - 1)
