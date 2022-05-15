# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import os
import typing
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from maro.simulator.scenarios.supply_chain.facilities.facility import FacilityInfo
from maro.simulator.scenarios.supply_chain.objects import SkuMeta, SupplyChainEntity
from maro.simulator.scenarios.supply_chain.units.consumer import ConsumerUnit
from maro.simulator.scenarios.supply_chain.units.manufacture import ManufactureUnit

if typing.TYPE_CHECKING:
    from examples.supply_chain.rl.env_sampler import SCEnvSampler

class SimulationTracker:
    def __init__(
        self, episode_len: int, n_episods: int, env_sampler: SCEnvSampler,
        log_path: str, eval_period: Optional[Tuple[int, int]] = None,
    ):
        self.loc_path = log_path
        os.makedirs(self.loc_path, exist_ok=True)

        self.episode_len = episode_len
        self.n_episods = n_episods

        if eval_period:
            self.eval_period = eval_period
        else:
            self.eval_period = [0, self.episode_len]

        self._env_sampler: SCEnvSampler = env_sampler
        self._facility_info_dict: Dict[int, FacilityInfo] = env_sampler._facility_info_dict
        self.entity_dict: Dict[int, SupplyChainEntity] = self._env_sampler._entity_dict
        self._sku_metas: Dict[int, SkuMeta] = env_sampler._sku_metas

        self._num_facilities: int = len(self._facility_info_dict)

        self.global_balances = np.zeros((n_episods, episode_len))
        self.global_rewards = np.zeros((n_episods, episode_len))

        self.step_balances = np.zeros((n_episods, self.episode_len, self._num_facilities))
        self.step_rewards = np.zeros((n_episods, self.episode_len, self._num_facilities))

        self.tracking_entity_ids: List[int] = []
        self.entity_id2idx_in_entity_id_list: Dict[int, int] = {}

        self.stock_status = None
        self.stock_in_transit_status = None
        self.order_to_distribute = None
        self.demand_status = None
        self.sold_status = None
        self.reward_status = None
        self.reward_discount_status = None

    def add_balance_and_reward(
        self, episode: int, t: int, global_balance: float, global_reward: float,
        step_balances: Dict[int, float], step_rewards: Dict[int, float]
    ):
        self.global_balances[episode, t] = global_balance
        self.global_rewards[episode, t] = global_reward
        for i, facility_id in enumerate(self._facility_info_dict.keys()):
            self.step_balances[episode, t, i] = step_balances[facility_id]
            self.step_rewards[episode, t, i] = step_rewards[facility_id]

    def add_sku_status(
        self, episode: int, t: int,
        stock: Dict[int, int], order_in_transit: Dict[int, int],
        demands: Dict[int, Union[int, float]], solds: Dict[int, Union[int, float]],
        rewards: Dict[int, float], balances: Dict[int, float],
        order_to_distribute: Dict[int, int],
    ):
        if len(self.tracking_entity_ids) == 0:
            # TODO: can refine tracking entity ids here.
            self.tracking_entity_ids = list(rewards.keys())
            for i, entity_id in enumerate(self.tracking_entity_ids):
                self.entity_id2idx_in_entity_id_list[entity_id] = i

            sku_status_shape = (self.n_episods, self.episode_len, len(self.tracking_entity_ids))

            self.stock_status = np.zeros(sku_status_shape)
            self.stock_in_transit_status = np.zeros(sku_status_shape)
            self.demand_status = np.zeros(sku_status_shape)
            self.sold_status = np.zeros(sku_status_shape)
            self.reward_status = np.zeros(sku_status_shape)
            self.balance_status = np.zeros(sku_status_shape)
            self.order_to_distribute = np.zeros(sku_status_shape)

        for i, entity_id in enumerate(self.tracking_entity_ids):
            self.stock_status[episode, t, i] = stock.get(entity_id, 0)
            self.stock_in_transit_status[episode, t, i] = order_in_transit.get(entity_id, 0)
            self.demand_status[episode, t, i] = demands.get(entity_id, 0)
            self.sold_status[episode, t, i] = solds.get(entity_id, 0)
            self.reward_status[episode, t, i] = rewards.get(entity_id, 0)
            self.balance_status[episode, t, i] = balances.get(entity_id, 0)
            self.order_to_distribute[episode, t, i] = order_to_distribute.get(entity_id, 0)

    def render_sku(self):
        for i, entity_id in enumerate(self.tracking_entity_ids):
            entity = self.entity_dict[entity_id]
            if not issubclass(entity.class_type, (ConsumerUnit, ManufactureUnit)):
                continue

            # TODO: only data in eval period used, no need to update before eval period.
            fig, ax = plt.subplots(3, 1, figsize=(25, 10))
            x = np.linspace(0, self.episode_len, self.episode_len)[self.eval_period[0]:self.eval_period[1]]
            stock = self.stock_status[0, :, i][self.eval_period[0]:self.eval_period[1]]
            order_in_transit = self.stock_in_transit_status[0, :, i][self.eval_period[0]:self.eval_period[1]]
            demand = self.demand_status[0, :, i][self.eval_period[0]:self.eval_period[1]]
            sold = self.sold_status[0, :, i][self.eval_period[0]:self.eval_period[1]]
            reward = self.reward_status[0, :, i][self.eval_period[0]:self.eval_period[1]]
            balance = self.balance_status[0, :, i][self.eval_period[0]:self.eval_period[1]]
            order_to_distribute = self.order_to_distribute[0, :, i][self.eval_period[0]:self.eval_period[1]]

            ax[0].set_title('SKU Stock Status by Episode')
            for y_label, y in [
                ('stock', stock),
                ('order_in_transit', order_in_transit),
                ('order_to_distribute', order_to_distribute),
            ]:
                ax[0].plot(x, y, label=y_label)

            ax[1].set_title('SKU Reward / Balance Status by Episode')
            ax[1].plot(x, balance, label='Balance')
            ax_r = ax[1].twinx()
            ax_r.plot(x, reward, label='Reward', color='r')

            ax[2].set_title('SKU demand')
            ax[2].plot(x, demand, label="Demand")
            ax[2].plot(x, sold, label="Sold")

            fig.legend()

            facility_info = self._facility_info_dict[entity.facility_id]
            file_name = (
                f"{facility_info.class_name.__name__}_"
                f"{facility_info.name}_"
                f"{self._sku_metas[entity.skus.id].name}_"
                f"{entity.id}"
                ".png"
            )

            fig.savefig(os.path.join(self.loc_path, file_name))
            plt.close(fig=fig)

    def render(self, file_name, metrics, facility_types):
        fig, axs = plt.subplots(2, 1, figsize=(25, 10))
        x = np.linspace(0, self.episode_len, self.episode_len)[self.eval_period[0]:self.eval_period[1]]

        _agent_list = []
        _step_idx = []
        for i, (facility_id, facility_info) in enumerate(self._facility_info_dict.items()):
            if facility_info.class_name.__name__ not in facility_types:
                continue
            _agent_list.append(f"{facility_info.name}_{facility_id}")
            _step_idx.append(i)
        _step_metrics = [metrics[0, self.eval_period[0]:self.eval_period[1], i] for i in _step_idx]

        # axs[0].set_title('Global balance')
        # axs[0].plot(x, self.global_balances.T)
        total_val = np.sum(np.sum(_step_metrics, axis=0))
        axs[0].set_title(f'Cumulative Sum_{total_val}')
        axs[0].plot(x, np.cumsum(np.sum(_step_metrics, axis=0)))

        axs[1].set_title('Breakdown by Agent (One Episode)')
        axs[1].plot(x, np.cumsum(_step_metrics, axis=1).T)
        axs[1].legend(_agent_list, loc='upper left')

        fig.savefig(f"{self.loc_path}/{file_name}")
        plt.close(fig=fig)

    def run_wth_render(self, facility_types):
        self._env_sampler.reset()
        self._env_sampler.start()
        for epoch in range(self.episode_len):
            states = self._env_sampler.get_state(None)
            action = {id_: self.learner.policy[id_].choose_action(st) for id_, st in states.items()}
            self._env_sampler.step(action)
            self._env_sampler.get_reward()
            step_balances = self._env_sampler.balance_status
            step_rewards = self._env_sampler.reward_status

            self.add_balance_and_reward(
                0, epoch, sum(step_balances.values()), sum(step_rewards.values()), step_balances, step_rewards
            )
            stock_status = self._env_sampler.stock_status
            order_in_transit_status = self._env_sampler.order_in_transit_status
            demand_status = self._env_sampler.demand_status
            sold_status = self._env_sampler.sold_status
            reward_status = self._env_sampler.reward_status
            balance_status = self._env_sampler.balance_status
            order_to_distribute_status = self._env_sampler.order_to_distribute_status

            self.add_sku_status(
                0, epoch, stock_status, order_in_transit_status, demand_status, sold_status,
                reward_status, balance_status, order_to_distribute_status
            )

        _step_idx = []
        for i, facility_info in enumerate(self._facility_info_dict.values()):
            if facility_info.class_name.__name__ in facility_types:
                _step_idx.append(i)

        _step_metrics = [self.step_rewards[0, self.eval_period[0]:self.eval_period[1], i] for i in _step_idx]
        _step_metrics_list = np.cumsum(np.sum(_step_metrics, axis=0))

        return np.sum(_step_metrics), _step_metrics_list

    def run_and_render(self, facility_types):
        metric, metric_list = self.run_wth_render(facility_types=facility_types)
        self.render('a_plot_balance.png', self.step_balances, ["StoreProductUnit"])
        self.render('a_plot_reward.png', self.step_rewards, ["StoreProductUnit"])
        self.render_sku()
        return metric, metric_list
