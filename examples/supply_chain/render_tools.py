# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import matplotlib.pyplot as plt
import numpy as np
import os

from examples.supply_chain.env_wrapper import sku_agent_types, SCEnvWrapper


class SimulationTracker:
    def __init__(self, episode_len: int, n_episodes: int, env: SCEnvWrapper, learner):
        self.episode_len = episode_len
        self.n_episodes = n_episodes
        self.env = env
        self.learner = learner

        self.global_balances = np.zeros((n_episodes, episode_len))
        self.global_rewards = np.zeros((n_episodes, episode_len))

        self.facility_names = []
        for agent in self.env._agent_list:
            if agent.agent_type in sku_agent_types:
                self.facility_names.append(agent)

        self.step_balances = np.zeros(
            (n_episodes, self.episode_len, len(self.facility_names)))
        self.step_rewards = np.zeros(
            (n_episodes, self.episode_len, len(self.facility_names)))

        self.sku_to_track = None
        self.stock_status = None
        self.stock_in_transit_status = None
        self.reward_status = None
        self.demand_status = None
        self.reward_discount_status = None
        self.order_to_distribute = None

    def _add_sample(self, episode, t, global_balance, global_reward, step_balances, step_rewards):
        self.global_balances[episode, t] = global_balance
        self.global_rewards[episode, t] = global_reward
        for i, f in enumerate(self.facility_names):
            self.step_balances[episode, t, i] = step_balances[f.id]
            self.step_rewards[episode, t, i] = step_rewards[f.id]

    def _add_sku_status(self, episode, t, stock, order_in_transit, demands, rewards, balances, order_to_distribute):
        if self.sku_to_track is None:
            self.sku_to_track = list(rewards.keys())
            self.stock_status = np.zeros(
                (self.n_episodes, self.episode_len, len(self.sku_to_track)))
            self.stock_in_transit_status = np.zeros(
                (self.n_episodes, self.episode_len, len(self.sku_to_track)))
            self.demand_status = np.zeros(
                (self.n_episodes, self.episode_len, len(self.sku_to_track)))
            self.reward_status = np.zeros(
                (self.n_episodes, self.episode_len, len(self.sku_to_track)))
            self.balance_status = np.zeros(
                (self.n_episodes, self.episode_len, len(self.sku_to_track)))
            self.order_to_distribute = np.zeros(
                (self.n_episodes, self.episode_len, len(self.sku_to_track)))
        for i, sku_name in enumerate(self.sku_to_track):
            self.stock_status[episode, t, i] = stock[sku_name]
            self.stock_in_transit_status[episode, t, i] = order_in_transit[sku_name]
            self.demand_status[episode, t, i] = demands[sku_name]
            self.reward_status[episode, t, i] = rewards[sku_name]
            self.balance_status[episode, t, i] = balances[sku_name]
            self.order_to_distribute[episode, t, i] = order_to_distribute[sku_name]

    def _render_sku(self, loc_path):
        sku_name_dict = {}
        facility_type_dict = {}
        for agent in self.env._agent_list:
            if agent.is_facility:
                sku_name = f"{agent.facility_id}_{agent.agent_type}"
            else:
                sku_name = f"{agent.id}_{agent.sku.id}_{agent.agent_type}"
            sku_name_dict[agent.id] = sku_name
            facility_type_dict[agent.id] = self.env.env.summary["node_mapping"]["facilities"][agent.facility_id]['class'].__name__

        for i, sku_name in enumerate(self.sku_to_track):
            fig, ax = plt.subplots(4, 1, figsize=(25, 10))
            x = np.linspace(0, self.episode_len, self.episode_len)
            stock = self.stock_status[0, :, i]
            order_in_transit = self.stock_in_transit_status[0, :, i]
            demand = self.demand_status[0, :, i]
            reward = self.reward_status[0, :, i]
            balance = self.balance_status[0, :, i]
            order_to_distribute = self.order_to_distribute[0, :, i]
            ax[0].set_title('SKU Stock Status by Episode')
            for y_label, y in [('stock', stock),
                               ('order_in_transit', order_in_transit),
                               ('order_to_distribute', order_to_distribute)]:
                ax[0].plot(x, y, label=y_label)

            ax[1].set_title('SKU Reward / Balance Status by Episode')
            ax[1].plot(x, balance, label='Balance')
            ax_r = ax[1].twinx()
            ax_r.plot(x, reward, label='Reward', color='r')

            ax[3].set_title('SKU demand')
            ax[3].plot(x, demand, label="Demand")

            fig.legend()
            fig.savefig(os.path.join(loc_path, f"{facility_type_dict[sku_name]}_{sku_name_dict[sku_name]}.png"))
            plt.close(fig=fig)

    def _render(self, file_name, metrics, facility_types):
        fig, axs = plt.subplots(2, 1, figsize=(25, 10))
        x = np.linspace(0, self.episode_len, self.episode_len)

        _agent_list = []
        _step_idx = []
        for i, agent_info in enumerate(self.facility_names):
            if agent_info.agent_type in ['productstore']:
                _agent_list.append(agent_info.id)
                _step_idx.append(i)
        _step_metrics = [metrics[0, :, i] for i in _step_idx]

        # axs[0].set_title('Global balance')
        # axs[0].plot(x, self.global_balances.T)

        axs[0].set_title('Cumulative Sum')
        axs[0].plot(x, np.cumsum(np.sum(_step_metrics, axis=0)))

        axs[1].set_title('Breakdown by Agent (One Episode)')
        axs[1].plot(x, np.cumsum(_step_metrics, axis=1).T)
        axs[1].legend(_agent_list, loc='upper left')

        fig.savefig(file_name)
        plt.close(fig=fig)
        # plt.show()

    def _run_wth_render(self, facility_types):
        self.env.reset()
        self.env.start()
        for epoch in range(self.episode_len):
            action = {id_: self.learner._policy[id_].choose_action(st) for id_, st in self.env.state.items()}
            self.learner._logger.info(f"epoch: {epoch}, action: {action}")
            self.env.step(action)
            self.learner._logger.info(f"epoch: {epoch}, action: {self.env.to_env_action(action)}")
            if hasattr(self.env, "consumer2product"):
                self.learner._logger.info(f"consumer2product: {self.env.consumer2product}")
            self.env.get_reward()
            step_balances = self.env.balance_status
            step_rewards = self.env.reward_status

            self._add_sample(0, epoch, sum(step_balances.values()), sum(
                step_rewards.values()), step_balances, step_rewards)
            stock_status = self.env.stock_status
            order_in_transit_status = self.env.order_in_transit_status
            demand_status = self.env.demand_status
            reward_status = self.env.reward_status
            balance_status = self.env.balance_status
            order_to_distribute_status = self.env.order_to_distribute_status

            self._add_sku_status(0, epoch, stock_status,
                                order_in_transit_status, demand_status,
                                reward_status, balance_status,
                                order_to_distribute_status)

        _step_idx = []
        for i, agent_info in enumerate(self.facility_names):
            if agent_info.agent_type in facility_types:
                _step_idx.append(i)
        _step_metrics = [self.step_rewards[0, :, i] for i in _step_idx]
        _step_metrics_list = np.cumsum(np.sum(_step_metrics, axis=0))
        return np.sum(_step_metrics), _step_metrics_list

    def run_and_render(self, loc_path, facility_types):
        if not os.path.exists(loc_path):
            os.makedirs(loc_path)
        metric, metric_list = self._run_wth_render(facility_types=facility_types)
        self._render(os.path.join(loc_path, "plot_balance.png"), self.step_balances, facility_types)
        self._render(os.path.join(loc_path, "plot_reward.png"), self.step_rewards, facility_types)
        self._render_sku(loc_path)
        return metric, metric_list
