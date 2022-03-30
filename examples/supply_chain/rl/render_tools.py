from maro.simulator.scenarios.supply_chain.units.product import ProductUnit
from maro.simulator.scenarios.supply_chain.facilities.facility import FacilityBase
import numpy as np
import matplotlib.pyplot as plt
import os


class SimulationTracker:
    def __init__(self, episod_len, n_episods, env, eval_period=None):
        self.episod_len = episod_len
        if eval_period:
            self.eval_period = eval_period
        else:
            self.eval_period = [0, self.episod_len]
        self.global_balances = np.zeros((n_episods, episod_len))
        self.global_rewards = np.zeros((n_episods, episod_len))
        self.env = env
        self.facility_info = env._summary['facilities']
        self.sku_meta_info = env._summary['skus']

        self.facility_names = []
        self.entity_dict = self.env._entity_dict
        for entity_id, entity in self.entity_dict.items():
            if entity.is_facility:
                self.facility_names.append(entity_id)

        self.step_balances = np.zeros(
            (n_episods, self.episod_len, len(self.facility_names)))
        self.step_rewards = np.zeros(
            (n_episods, self.episod_len, len(self.facility_names)))
        self.n_episods = n_episods
        self.sku_to_track = None
        self.stock_status = None
        self.stock_in_transit_status = None
        self.reward_status = None
        self.sold_status = None
        self.demand_status = None
        self.reward_discount_status = None
        self.order_to_distribute = None

    def add_sample(self, episod, t, global_balance, global_reward, step_balances, step_rewards):
        self.global_balances[episod, t] = global_balance
        self.global_rewards[episod, t] = global_reward
        for i, f_id in enumerate(self.facility_names):
            self.step_balances[episod, t, i] = step_balances.get(f_id, 0)
            self.step_rewards[episod, t, i] = step_rewards.get(f_id, 0)

    def add_sku_status(self, episod, t, stock, order_in_transit, demands, solds, rewards, balances, order_to_distribute):
        if self.sku_to_track is None:
            self.sku_to_track = list(rewards.keys())
            self.stock_status = np.zeros(
                (self.n_episods, self.episod_len, len(self.sku_to_track)))
            self.stock_in_transit_status = np.zeros(
                (self.n_episods, self.episod_len, len(self.sku_to_track)))
            self.demand_status = np.zeros(
                (self.n_episods, self.episod_len, len(self.sku_to_track)))
            self.sold_status = np.zeros(
                (self.n_episods, self.episod_len, len(self.sku_to_track)))
            self.reward_status = np.zeros(
                (self.n_episods, self.episod_len, len(self.sku_to_track)))
            self.balance_status = np.zeros(
                (self.n_episods, self.episod_len, len(self.sku_to_track)))
            self.order_to_distribute = np.zeros(
                (self.n_episods, self.episod_len, len(self.sku_to_track)))
        for i, sku_name in enumerate(self.sku_to_track):
            self.stock_status[episod, t, i] = stock.get(sku_name, 0)
            self.stock_in_transit_status[episod,
                                         t, i] = order_in_transit.get(sku_name, 0)
            self.demand_status[episod, t, i] = demands.get(sku_name, 0)
            self.sold_status[episod, t, i] = solds.get(sku_name, 0)
            self.reward_status[episod, t, i] = rewards.get(sku_name, 0)
            self.balance_status[episod, t, i] = balances.get(sku_name, 0)
            self.order_to_distribute[episod, t,
                                     i] = order_to_distribute.get(sku_name, 0)

    def render_sku(self, loc_path):
        sku_name_dict = {}
        facility_type_dict = {}
        for entity_id in self.sku_to_track:
            entity = self.entity_dict[entity_id]
            if not issubclass(entity.class_type, ProductUnit):
                continue
            facility = self.facility_info[entity.facility_id]
            sku = self.sku_meta_info[entity.skus.id]
            sku_name = f"{facility['name']}_{sku.name}_{entity.id}"
            facility_type_dict[entity.id] = facility['class'].__name__
            sku_name_dict[entity.id] = sku_name

        for i, entity_id in enumerate(self.sku_to_track):
            entity = self.entity_dict[entity_id]
            if not issubclass(entity.class_type, ProductUnit):
                continue
            fig, ax = plt.subplots(4, 1, figsize=(25, 10))
            x = np.linspace(0, self.episod_len, self.episod_len)[self.eval_period[0]:self.eval_period[1]]
            stock = self.stock_status[0, :, i][self.eval_period[0]:self.eval_period[1]]
            order_in_transit = self.stock_in_transit_status[0, :, i][self.eval_period[0]:self.eval_period[1]]
            demand = self.demand_status[0, :, i][self.eval_period[0]:self.eval_period[1]]
            sold =  self.sold_status[0, :, i][self.eval_period[0]:self.eval_period[1]]
            reward = self.reward_status[0, :, i][self.eval_period[0]:self.eval_period[1]]
            balance = self.balance_status[0, :, i][self.eval_period[0]:self.eval_period[1]]
            order_to_distribute = self.order_to_distribute[0, :, i][self.eval_period[0]:self.eval_period[1]]

            ax[0].set_title('SKU Stock Status by Episod')
            for y_label, y in [('stock', stock),
                               ('order_in_transit', order_in_transit),
                               ('order_to_distribute', order_to_distribute)]:
                ax[0].plot(x, y, label=y_label)

            ax[1].set_title('SKU Reward / Balance Status by Episod')
            ax[1].plot(x, balance, label='Balance')
            ax_r = ax[1].twinx()
            ax_r.plot(x, reward, label='Reward', color='r')

            ax[3].set_title('SKU demand')
            ax[3].plot(x, demand, label="Demand")
            ax[3].plot(x, sold, label="Sold")

            fig.legend()
            fig.savefig(f"{loc_path}/{facility_type_dict[entity_id]}_{sku_name_dict[entity_id]}.png")
            plt.close(fig=fig)

    def render(self, file_name, metrics, facility_types):
        fig, axs = plt.subplots(2, 1, figsize=(25, 10))
        x = np.linspace(0, self.episod_len, self.episod_len)[self.eval_period[0]:self.eval_period[1]]

        _agent_list = []
        _step_idx = []
        for i, entity_id in enumerate(self.facility_names):
            entity = self.entity_dict[entity_id]
            if not (entity.class_type.__name__ in facility_types):
                continue
            facility = self.facility_info[entity_id]
            _agent_list.append(f"{facility['name']}_{entity_id}")
            _step_idx.append(i)
        _step_metrics = [metrics[0, self.eval_period[0]:self.eval_period[1], i] for i in _step_idx]

        # axs[0].set_title('Global balance')
        # axs[0].plot(x, self.global_balances.T)

        axs[0].set_title('Cumulative Sum')
        axs[0].plot(x, np.cumsum(np.sum(_step_metrics, axis=0)))

        axs[1].set_title('Breakdown by Agent (One Episod)')
        axs[1].plot(x, np.cumsum(_step_metrics, axis=1).T)
        axs[1].legend(_agent_list, loc='upper left')

        fig.savefig(file_name)
        plt.close(fig=fig)
        # plt.show()

    def run_wth_render(self, facility_types):
        self.env.reset()
        self.env.start()
        for epoch in range(self.episod_len):
            states = self.env.get_state(None)
            action = {id_: self.learner.policy[id_].choose_action(st) for id_, st in states.items()}
            self.env.step(action)
            self.env.get_reward()
            step_balances = self.env.balance_status
            step_rewards = self.env.reward_status

            self.add_sample(0, epoch, sum(step_balances.values()), sum(
                step_rewards.values()), step_balances, step_rewards)
            stock_status = self.env.stock_status
            order_in_transit_status = self.env.order_in_transit_status
            demand_status = self.env.demand_status
            sold_status = self.env.sold_status
            reward_status = self.env.reward_status
            balance_status = self.env.balance_status
            order_to_distribute_status = self.env.order_to_distribute_status

            self.add_sku_status(0, epoch, stock_status,
                                order_in_transit_status, demand_status, sold_status,
                                reward_status, balance_status,
                                order_to_distribute_status)

        _step_idx = []
        for i, entity_id in enumerate(self.facility_names):
            if self.entity_dict[entity_id].class_type.__name__ in facility_types:
                _step_idx.append(i)
        _step_metrics = [self.step_rewards[0, self.eval_period[0]:self.eval_period[1], i] for i in _step_idx]
        _step_metrics_list = np.cumsum(np.sum(_step_metrics, axis=0))
        return np.sum(_step_metrics), _step_metrics_list

    def run_and_render(self, loc_path, facility_types):
        metric, metric_list = self.run_wth_render(
            facility_types=facility_types)
        self.render('%s/a_plot_balance.png' %
                    loc_path, self.step_balances, ["StoreProductUnit"])
        self.render('%s/a_plot_reward.png' %
                    loc_path, self.step_rewards, ["StoreProductUnit"])
        self.render_sku(loc_path)
        return metric, metric_list
