# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

import matplotlib.pyplot as plt
import numpy as np

from maro.simulator.scenarios.supply_chain.facilities.facility import FacilityBase
from maro.simulator.scenarios.supply_chain.units.product import ProductUnit


class SimulationTracker:
    def __init__(self, episode_len, n_episods, env, log_path, eval_period=None):
        self.loc_path = log_path
        os.makedirs(self.loc_path, exist_ok=True)

        self.episode_len = episode_len

        if eval_period:
            self.eval_period = eval_period
        else:
            self.eval_period = [0, self.episode_len]

        self.global_balances = np.zeros((n_episods, episode_len))
        self.global_rewards = np.zeros((n_episods, episode_len))

        self.env = env
        self.facility_info = env._summary['facilities']
        self.sku_meta_info = env._summary['skus']

        self.facility_names = []
        self.entity_dict = self.env._entity_dict
        for entity_id, entity in self.entity_dict.items():
            if issubclass(entity.class_type, FacilityBase):
                self.facility_names.append(entity_id)

        self.step_balances = np.zeros((n_episods, self.episode_len, len(self.facility_names)))
        self.step_rewards = np.zeros((n_episods, self.episode_len, len(self.facility_names)))
        self.n_episods = n_episods
        self.sku_to_track = None
        self.stock_status = None
        self.stock_in_transit_status = None
        self.reward_status = None
        self.sold_status = None
        self.demand_status = None
        self.reward_discount_status = None
        self.order_to_distribute = None

    def add_sample(self, episode, t, global_balance, global_reward, step_balances, step_rewards):
        self.global_balances[episode, t] = global_balance
        self.global_rewards[episode, t] = global_reward
        for i, f_id in enumerate(self.facility_names):
            self.step_balances[episode, t, i] = step_balances.get(f_id, 0)
            self.step_rewards[episode, t, i] = step_rewards.get(f_id, 0)

    def add_sku_status(
        self, episode, t, stock, order_in_transit, demands, solds, rewards, balances, order_to_distribute
    ):
        if self.sku_to_track is None:
            self.sku_to_track = list(rewards.keys())

            sku_status_shape = (self.n_episods, self.episode_len, len(self.sku_to_track))
            self.stock_status = np.zeros(sku_status_shape)
            self.stock_in_transit_status = np.zeros(sku_status_shape)
            self.demand_status = np.zeros(sku_status_shape)
            self.sold_status = np.zeros(sku_status_shape)
            self.reward_status = np.zeros(sku_status_shape)
            self.balance_status = np.zeros(sku_status_shape)
            self.order_to_distribute = np.zeros(sku_status_shape)

        for i, sku_name in enumerate(self.sku_to_track):
            self.stock_status[episode, t, i] = stock.get(sku_name, 0)
            self.stock_in_transit_status[episode, t, i] = order_in_transit.get(sku_name, 0)
            self.demand_status[episode, t, i] = demands.get(sku_name, 0)
            self.sold_status[episode, t, i] = solds.get(sku_name, 0)
            self.reward_status[episode, t, i] = rewards.get(sku_name, 0)
            self.balance_status[episode, t, i] = balances.get(sku_name, 0)
            self.order_to_distribute[episode, t, i] = order_to_distribute.get(sku_name, 0)

    def render_sku(self):
        sku_name_dict = {}
        facility_type_dict = {}
        for entity_id in self.sku_to_track:
            entity = self.entity_dict[entity_id]
            if not issubclass(entity.class_type, ProductUnit):
                continue
            facility = self.facility_info[entity.facility_id]
            sku = self.sku_meta_info[entity.skus.id]
            sku_name = f"{facility.name}_{sku.name}_{entity.id}"
            facility_type_dict[entity.id] = facility.class_name.__name__
            sku_name_dict[entity.id] = sku_name

        for i, entity_id in enumerate(self.sku_to_track):
            entity = self.entity_dict[entity_id]
            if not issubclass(entity.class_type, ProductUnit):
                continue
            facility = self.facility_info[entity.facility_id]
            if facility.name.startswith("VNDR"):
                continue
            fig, ax = plt.subplots(4, 1, figsize=(25, 10))
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

            ax[3].set_title('SKU demand')
            ax[3].plot(x, demand, label="Demand")
            ax[3].plot(x, sold, label="Sold")

            fig.legend()
            fig.savefig(f"{self.loc_path}/{facility_type_dict[entity_id]}_{sku_name_dict[entity_id]}.png")
            plt.close(fig=fig)

    def render(self, file_name, metrics, facility_types):
        fig, axs = plt.subplots(2, 1, figsize=(25, 10))
        x = np.linspace(0, self.episode_len, self.episode_len)[self.eval_period[0]:self.eval_period[1]]

        _agent_list = []
        _step_idx = []
        for i, entity_id in enumerate(self.facility_names):
            entity = self.entity_dict[entity_id]
            if not (entity.class_type.__name__ in facility_types):
                continue
            facility = self.facility_info[entity_id]
            _agent_list.append(f"{facility.name}_{entity_id}")
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
        self.env.reset()
        self.env.start()
        for epoch in range(self.episode_len):
            states = self.env.get_state(None)
            action = {id_: self.learner.policy[id_].choose_action(st) for id_, st in states.items()}
            self.env.step(action)
            self.env.get_reward()
            step_balances = self.env.balance_status
            step_rewards = self.env.reward_status

            self.add_sample(
                0, epoch, sum(step_balances.values()), sum(step_rewards.values()), step_balances, step_rewards
            )
            stock_status = self.env.stock_status
            order_in_transit_status = self.env.order_in_transit_status
            demand_status = self.env.demand_status
            sold_status = self.env.sold_status
            reward_status = self.env.reward_status
            balance_status = self.env.balance_status
            order_to_distribute_status = self.env.order_to_distribute_status

            self.add_sku_status(
                0, epoch, stock_status, order_in_transit_status, demand_status, sold_status,
                reward_status, balance_status, order_to_distribute_status
            )

        _step_idx = []
        for i, entity_id in enumerate(self.facility_names):
            if self.entity_dict[entity_id].class_type.__name__ in facility_types:
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
