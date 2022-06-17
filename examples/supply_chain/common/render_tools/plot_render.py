# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import os
import pickle
import typing
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from maro.simulator.scenarios.supply_chain.facilities.facility import FacilityInfo
from maro.simulator.scenarios.supply_chain.objects import SkuMeta, SupplyChainEntity

if typing.TYPE_CHECKING:
    from examples.supply_chain.rl.env_sampler import SCEnvSampler


class SimulationTracker:
    def __init__(
        self,
        episode_len: int,
        n_episodes: int,
        env_sampler: SCEnvSampler,
        log_path: str,
        eval_period: Optional[Tuple[int, int]] = None,
    ):
        self._log_path = log_path
        os.makedirs(self._log_path, exist_ok=True)

        self._sku_render_path = os.path.join(self._log_path, "sku_status")
        os.makedirs(self._sku_render_path, exist_ok=True)

        self._episode_len = episode_len
        self._n_episodes = n_episodes

        if eval_period:
            self.eval_period = eval_period
            assert self.eval_period[0] >= 0
            assert self.eval_period[1] <= self._episode_len
            assert self.eval_period[0] < self.eval_period[1]
        else:
            self.eval_period = [0, self._episode_len]

        self._env_sampler: SCEnvSampler = env_sampler
        self._facility_info_dict: Dict[int, FacilityInfo] = env_sampler._facility_info_dict
        self._entity_dict: Dict[int, SupplyChainEntity] = self._env_sampler._entity_dict
        self._sku_metas: Dict[int, SkuMeta] = env_sampler._sku_metas

        self._num_facilities: int = len(self._facility_info_dict)

        self._n_eval_steps = self.eval_period[1] - self.eval_period[0]
        self.global_balances: np.ndarray = np.zeros((n_episodes, self._n_eval_steps))
        self.global_rewards: np.ndarray = np.zeros((n_episodes, self._n_eval_steps))

        self.step_balances: np.ndarray = np.zeros((n_episodes, self._n_eval_steps, self._num_facilities))
        self.step_rewards: np.ndarray = np.zeros((n_episodes, self._n_eval_steps, self._num_facilities))

        self.tracking_entity_ids: List[int] = []
        self._entity_id2idx_in_status: Dict[int, int] = {}

        self.stock_status: np.ndarray = None
        self.stock_in_transit_status: np.ndarray = None
        self.stock_ordered_to_distribute_status: np.ndarray = None

        self.demand_status: np.ndarray = None
        self.sold_status: np.ndarray = None

        self.reward_status: np.ndarray = None
        self.balance_status: np.ndarray = None

        self.consumer_purchased: np.ndarray = None
        self.consumer_received: np.ndarray = None
        self.manufacture_started: np.ndarray = None
        self.manufacture_finished: np.ndarray = None

    def add_balance_and_reward(
        self,
        episode: int,
        tick: int,
        global_balance: float,
        global_reward: float,
        step_balances: Dict[int, float],
        step_rewards: Dict[int, float],
    ):
        if tick < self.eval_period[0] or tick >= self.eval_period[1]:
            return
        t = tick - self.eval_period[0]

        self.global_balances[episode, t] = global_balance
        self.global_rewards[episode, t] = global_reward
        for i, facility_id in enumerate(self._facility_info_dict.keys()):
            self.step_balances[episode, t, i] = step_balances[facility_id]
            self.step_rewards[episode, t, i] = step_rewards[facility_id]

    def add_sku_status(
        self,
        episode: int,
        tick: int,
        stock: Dict[int, int],
        stock_in_transit: Dict[int, int],
        stock_ordered_to_distribute: Dict[int, int],
        demands: Dict[int, Union[int, float]],
        solds: Dict[int, Union[int, float]],
        rewards: Dict[int, float],
        balances: Dict[int, float],
    ):
        if tick < self.eval_period[0] or tick >= self.eval_period[1]:
            return
        t = tick - self.eval_period[0]

        if len(self.tracking_entity_ids) == 0:
            # TODO: can refine tracking entity ids here.
            self.tracking_entity_ids = list(rewards.keys())
            self._init_sku_status()

        for i, entity_id in enumerate(self.tracking_entity_ids):
            self.stock_status[episode, t, i] = stock.get(entity_id, 0)
            self.stock_in_transit_status[episode, t, i] = stock_in_transit.get(entity_id, 0)
            self.stock_ordered_to_distribute_status[episode, t, i] = stock_ordered_to_distribute.get(entity_id, 0)
            self.demand_status[episode, t, i] = demands.get(entity_id, 0)
            self.sold_status[episode, t, i] = solds.get(entity_id, 0)
            self.reward_status[episode, t, i] = rewards.get(entity_id, 0)
            self.balance_status[episode, t, i] = balances.get(entity_id, 0)

    def add_action_status(
        self,
        consumer_purchased: np.ndarray,
        consumer_received: np.ndarray,
        manufacture_started: np.ndarray,
        manufacture_finished: np.ndarray,
    ) -> None:
        self.consumer_purchased = consumer_purchased
        self.consumer_received = consumer_received

        self.manufacture_started = manufacture_started
        self.manufacture_finished = manufacture_finished

    def _init_sku_status(self) -> None:
        for i, entity_id in enumerate(self.tracking_entity_ids):
            self._entity_id2idx_in_status[entity_id] = i

        sku_status_shape = (self._n_episodes, self._n_eval_steps, len(self.tracking_entity_ids))

        self.stock_status = np.zeros(sku_status_shape)
        self.stock_in_transit_status = np.zeros(sku_status_shape)
        self.stock_ordered_to_distribute_status = np.zeros(sku_status_shape)
        self.demand_status = np.zeros(sku_status_shape)
        self.sold_status = np.zeros(sku_status_shape)
        self.reward_status = np.zeros(sku_status_shape)
        self.balance_status = np.zeros(sku_status_shape)

    def dump_sku_status(self, entity_types: Tuple[type, ...]) -> None:
        dump_data = {
            "step_balance": self.step_balances,
            "step_reward": self.step_rewards,
            "facility_infos": [
                (
                    i,  # index in step_balance & step_reward
                    facility_id,
                    facility_info.name,
                    facility_info.class_name,
                )
                for i, (facility_id, facility_info) in enumerate(self._facility_info_dict.items())
            ],
            "entity_infos": [
                (
                    self._entity_id2idx_in_status[entity_id],  # index in status
                    entity_id,  # entity id
                    self._facility_info_dict[self._entity_dict[entity_id].facility_id].name,  # facility name
                    self._sku_metas[self._entity_dict[entity_id].skus.id].name,  # sku name
                    self._entity_dict[entity_id].class_type,  # entity class type
                )
                for entity_id in self.tracking_entity_ids
                if issubclass(self._entity_dict[entity_id].class_type, entity_types)
            ],
            "tracking_entity_ids": self.tracking_entity_ids,
            "stock_status": self.stock_status,
            "stock_in_transit_status": self.stock_in_transit_status,
            "stock_ordered_to_distribute_status": self.stock_ordered_to_distribute_status,
            "demand_status": self.demand_status,
            "sold_status": self.sold_status,
            "reward_status": self.reward_status,
            "balance_status": self.balance_status,
            "consumer_purchased": self.consumer_purchased,
            "consumer_received": self.consumer_received,
            "manufacture_started": self.manufacture_started,
            "manufacture_finished": self.manufacture_finished,
        }

        with open(os.path.join(self._log_path, "sku_status.pkl"), "wb") as fout:
            pickle.dump(dump_data, fout)

    def load_sku_status(self, file_path: str = None) -> None:
        if file_path is None:
            file_path = os.path.join(self._log_path, "sku_status.pkl")

        dump_data = {}
        with open(file_path, "rb") as fin:
            dump_data = pickle.load(fin)

        self.tracking_entity_ids = dump_data.get("tracking_entity_ids", [])
        self._entity_id2idx_in_status.clear()
        for i, entity_id in enumerate(self.tracking_entity_ids):
            self._entity_id2idx_in_status[entity_id] = i

        self.stock_status = dump_data.get("stock_status", None)
        self.stock_in_transit_status = dump_data.get("stock_in_transit_status", None)
        self.stock_ordered_to_distribute_status = dump_data.get("stock_ordered_to_distribute_status", None)
        self.demand_status = dump_data.get("demand_status", None)
        self.sold_status = dump_data.get("sold_status", None)
        self.reward_status = dump_data.get("reward_status", None)
        self.balance_status = dump_data.get("balance_status", None)

        if len(self.tracking_entity_ids) == 0:
            for variable in [
                self.stock_status,
                self.stock_in_transit_status,
                self.stock_ordered_to_distribute_status,
                self.demand_status,
                self.sold_status,
                self.reward_status,
                self.balance_status,
            ]:
                assert variable is None, f"Loaded Non-None status but with valid tracking entity id list!"

            print(f"[Warning] No sku status data loaded from {file_path}")
        else:
            assert isinstance(self.stock_status, np.ndarray), f"Status data must be np.ndarray!"
            assert self.stock_status.shape[2] == len(self.tracking_entity_ids), f"The last dim must be entity list len!"
            for variable in [
                self.stock_in_transit_status,
                self.stock_ordered_to_distribute_status,
                self.demand_status,
                self.sold_status,
                self.reward_status,
                self.balance_status,
            ]:
                assert isinstance(variable, np.ndarray), f"Status data must be np.ndarray!"
                assert variable.shape == self.stock_status.shape, f"Shapes must be the same!"

    def _render_sku(self, entity_id: int) -> None:
        entity = self._entity_dict[entity_id]
        i = self._entity_id2idx_in_status[entity_id]

        # TODO: only data in eval period used, no need to update before eval period.
        fig, ax = plt.subplots(3, 1, figsize=(25, 10))
        x = np.linspace(self.eval_period[0], self.eval_period[1] - 1, self._n_eval_steps)

        ax[0].set_title("SKU Stock Status by Episode")
        for y_label, y in [
            ("stock", self.stock_status[0, :, i]),
            ("stock_in_transit", self.stock_in_transit_status[0, :, i]),
            ("stock_to_distribute", self.stock_ordered_to_distribute_status[0, :, i]),
        ]:
            ax[0].plot(x, y, label=y_label)

        ax[1].set_title("SKU Reward / Balance Status by Episode")
        ax[1].plot(x, self.balance_status[0, :, i], label="Balance")
        ax_r = ax[1].twinx()
        ax_r.plot(x, self.reward_status[0, :, i], label="Reward", color="r")

        ax[2].set_title("SKU demand")
        ax[2].plot(x, self.demand_status[0, :, i], label="Demand")
        ax[2].plot(x, self.sold_status[0, :, i], label="Sold")

        fig.legend()

        facility_info = self._facility_info_dict[entity.facility_id]
        file_name = (
            # f"{facility_info.class_name.__name__}_"
            f"{facility_info.name}_"
            f"{self._sku_metas[entity.skus.id].name}_"
            # f"{entity.id}"
            ".png"
        )

        fig.savefig(os.path.join(self._sku_render_path, file_name))
        plt.close(fig=fig)

    def render_all_sku(self, entity_types: Tuple[type, ...] = None):
        for entity_id in self.tracking_entity_ids:
            entity = self._entity_dict[entity_id]
            if entity_types is None or issubclass(entity.class_type, entity_types):
                self._render_sku(entity_id)

    def _render_balance_or_reward(
        self,
        metrics: np.ndarray,
        data_idx_list: List[int],
        legend_name_list: List[str],
        filename: str,
    ) -> None:
        step_metrics = [metrics[0, :, i] for i in data_idx_list]
        total_val = np.sum(np.sum(step_metrics, axis=0))

        fig, axs = plt.subplots(2, 1, figsize=(25, 10))
        x = np.linspace(self.eval_period[0], self.eval_period[1] - 1, self._n_eval_steps)

        axs[0].set_title(f"Cumulative Sum_{total_val}")
        axs[0].plot(x, np.cumsum(np.sum(step_metrics, axis=0)))

        axs[1].set_title("Breakdown by Agent (One Episode)")
        axs[1].plot(x, np.cumsum(step_metrics, axis=1).T)
        axs[1].legend(legend_name_list, loc="upper left")

        fig.savefig(os.path.join(self._log_path, filename))
        plt.close(fig=fig)

    def render_facility_balance_and_reward(self, facility_types: Tuple[type, ...] = None) -> None:
        for metrics, filename in zip(
            [self.step_balances, self.step_rewards],
            ["facility_balance.png", "facility_reward.png"],
        ):
            data_idx_list = []
            legend_name_list = []
            for i, (facility_id, facility_info) in enumerate(self._facility_info_dict.items()):
                if facility_types is None or issubclass(facility_info.class_name, facility_types):
                    data_idx_list.append(i)
                    legend_name_list.append(f"{facility_info.name}_{facility_id}")
            self._render_balance_or_reward(metrics, data_idx_list, legend_name_list, filename)
