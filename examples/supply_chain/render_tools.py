import numpy as np
import matplotlib.pyplot as plt
import os


class SimulationTracker:
    def __init__(self, episod_len, n_episods, env, learner):
        self.episod_len = episod_len
        self.global_balances = np.zeros((n_episods, episod_len))
        self.global_rewards = np.zeros((n_episods, episod_len))
        self.env = env
        self.learner = learner
        self.facility_names = self.env._agent_list
        self.step_balances = np.zeros(
            (n_episods, self.episod_len, len(self.facility_names)))
        self.step_rewards = np.zeros(
            (n_episods, self.episod_len, len(self.facility_names)))
        self.n_episods = n_episods
        self.sku_to_track = None
        self.stock_status = None
        self.stock_in_transit_status = None
        self.reward_status = None
        self.demand_status = None
        self.reward_discount_status = None
        self.order_to_distribute = None

    def add_sample(self, episod, t, global_balance, global_reward, step_balances, step_rewards):
        self.global_balances[episod, t] = global_balance
        self.global_rewards[episod, t] = global_reward
        for i, f in enumerate(self.facility_names):
            self.step_balances[episod, t, i] = step_balances[f.id]
            self.step_rewards[episod, t, i] = step_rewards[f.id]

    def add_sku_status(self, episod, t, stock, order_in_transit, demands, rewards, balances, order_to_distribute):
        if self.sku_to_track is None:
            self.sku_to_track = set(
                list(stock.keys()) + list(order_in_transit.keys()) + list(demands.keys()))
            self.stock_status = np.zeros(
                (self.n_episods, self.episod_len, len(self.sku_to_track)))
            self.stock_in_transit_status = np.zeros(
                (self.n_episods, self.episod_len, len(self.sku_to_track)))
            self.demand_status = np.zeros(
                (self.n_episods, self.episod_len, len(self.sku_to_track)))
            self.reward_status = np.zeros(
                (self.n_episods, self.episod_len, len(self.sku_to_track)))
            self.balance_status = np.zeros(
                (self.n_episods, self.episod_len, len(self.sku_to_track)))
            self.order_to_distribute = np.zeros(
                (self.n_episods, self.episod_len, len(self.sku_to_track)))
        for i, sku_name in enumerate(self.sku_to_track):
            self.stock_status[episod, t, i] = stock[sku_name]
            self.stock_in_transit_status[episod,
                                         t, i] = order_in_transit[sku_name]
            self.demand_status[episod, t, i] = demands[sku_name]
            self.reward_status[episod, t, i] = rewards[sku_name]
            self.balance_status[episod, t, i] = balances[sku_name]
            self.order_to_distribute[episod, t,
                                     i] = order_to_distribute[sku_name]

    def render_sku(self, loc_path):
        for i, sku_name in enumerate(self.sku_to_track):
            fig, ax = plt.subplots(3, 1, figsize=(25, 10))
            x = np.linspace(0, self.episod_len, self.episod_len)
            stock = self.stock_status[0, :, i]
            order_in_transit = self.stock_in_transit_status[0, :, i]
            demand = self.demand_status[0, :, i]
            reward = self.reward_status[0, :, i]
            balance = self.balance_status[0, :, i]
            order_to_distribute = self.order_to_distribute[0, :, i]
            ax[0].set_title('SKU Stock Status by Episod')
            for y_label, y in [('stock', stock),
                               ('order_in_transit', order_in_transit),
                               ('demand', demand),
                               ('order_to_distribute', order_to_distribute)]:
                ax[0].plot(x, y, label=y_label)

            ax[1].set_title('SKU Reward / Balance Status by Episod')
            ax[1].plot(x, balance, label='Balance')
            ax_r = ax[1].twinx()
            ax_r.plot(x, reward, label='Reward', color='r')
            fig.legend()
            fig.savefig(f"{loc_path}/{sku_name}.png")
            plt.close(fig=fig)

    def render(self, file_name, metrics, facility_types):
        fig, axs = plt.subplots(2, 1, figsize=(25, 10))
        x = np.linspace(0, self.episod_len, self.episod_len)

        _agent_list = []
        _step_idx = []
        for i, agent_info in enumerate(self.facility_names):
            if agent_info.agent_type in facility_types:
                _agent_list.append(agent_info.id)
                _step_idx.append(i)
        _step_metrics = [metrics[0, :, i] for i in _step_idx]

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
        self.learner.policy.eval_mode()
        for epoch in range(self.episod_len):
            action = self.learner.policy.choose_action(self.env.state)
            self.learner._logger.info(f"epoch: {epoch}, action: {action}")
            self.env.step(action)
            self.learner._logger.info(f"epoch: {epoch}, action: {self.env.get_action(action)}")
            if hasattr(self.env, "consumer2product"):
                self.learner._logger.info(f"consumer2product: {self.env.consumer2product}")
            self.env.get_reward()
            step_balances = self.env.balance_status
            step_rewards = self.env.reward_status

            self.add_sample(0, epoch, sum(step_balances.values()), sum(
                step_rewards.values()), step_balances, step_rewards)
            stock_status = self.env.stock_status
            order_in_transit_status = self.env.order_in_transit_status
            demand_status = self.env.demand_status
            reward_status = self.env.reward_status
            balance_status = self.env.balance_status
            order_to_distribute_status = self.env.order_to_distribute_status

            self.add_sku_status(0, epoch, stock_status,
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
        metric, metric_list = self.run_wth_render(
            facility_types=facility_types)
        os.makedirs(loc_path, exist_ok=True)
        self.render('%s/plot_balance.png' %
                    loc_path, self.step_balances, facility_types)
        self.render('%s/plot_reward.png' %
                    loc_path, self.step_rewards, facility_types)
        self.render_sku(loc_path)
        return metric, metric_list
