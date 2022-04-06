from maro.simulator.scenarios.supply_chain.units.product import ProductUnit
from maro.simulator.scenarios.supply_chain.facilities.facility import FacilityBase
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pyecharts.options as opts
from pyecharts.charts import Line, Scatter, Bar, Timeline, Grid, Tab, Pie
from datetime import datetime, timedelta

"""
tick,id,sku_id,facility_id,
facility_name,name,inventory_in_stock,
unit_inventory_holding_cost,
consumer_purchased,
consumer_received,consumer_order_cost,
consumer_order_product_cost,seller_sold,seller_demand,
seller_price,seller_backlog_ratio,
manufacture_manufacture_quantity,
manufacture_product_unit_cost,product_price,
product_check_in_quantity_in_order,product_delay_order_penalty,
product_transportation_cost
"""

def compute_balance(row):
    return (row["seller_sold"]*row['seller_price']
           - (row["seller_demand"]-row['seller_sold'])*row['seller_price']*row['seller_backlog_ratio']
           - row["consumer_order_product_cost"]
           - row["consumer_order_cost"]
           - row['unit_inventory_holding_cost']*row['inventory_in_stock'])


class SimulationTrackerHtml:
    def __init__(self, log_path, start_dt='2022-01-01'):
        self.log_path = log_path
        self.start_dt = datetime.strptime(start_dt, "%Y-%m-%d")
        self.dir_loc = os.path.dirname(self.log_path)

    def render_facility(self):
        df_all = pd.read_csv(self.log_path)
        facility_list = df_all['facility_name'].unique().tolist()
        for facility_name in facility_list:
            df = df_all[df_all['facility_name'] == facility_name]
            df.loc[:, 'inventory_in_stock'] = df['inventory_in_stock'].map(lambda x: np.sum([int(float(s)) for s in x[1:-1].split(",")]))
            df.loc[:, 'unit_inventory_holding_cost'] = df['unit_inventory_holding_cost'].map(lambda x: np.sum([int(float(s)) for s in x[1:-1].split(",")]))
            df.loc[:, "profit"] = df.apply(lambda x: compute_balance(x), axis=1)
            if df.shape[0] > 0:
                df = df[['tick', 'facility_name', 'profit', 'inventory_in_stock']].groupby('tick').sum().reset_index()
                df.loc[:, 'sale_dt'] = df['tick'].map(lambda x: self.start_dt+timedelta(days=x))
                df.sort_values(by='sale_dt', inplace=True)
                df.loc[:, 'sale_dt'] = df['sale_dt'].map(lambda x: x.strftime('%Y-%m-%d'))
                x = df['sale_dt'].tolist()
                y_stock = [int(x) for x in df['inventory_in_stock'].tolist()]
                y_profit = [int(x) for x in df['profit'].tolist()]
            else:
                x = [self.start_dt.strftime('%Y-%m-%d')]
                y_stock = [0]
                y_profit = [0]
            y_profit_cum = np.cumsum(y_profit).tolist()
            tl = Timeline(init_opts=opts.InitOpts(width="1500px", height="800px"))
            tl.add_schema(pos_bottom="bottom", is_auto_play=False, label_opts=opts.LabelOpts(is_show=True, position="bottom"))
            file_name = os.path.join(self.dir_loc, f"{facility_name}.html")
            l1 = (
                Line()
                .add_xaxis(xaxis_data=x)
                .add_yaxis(
                    series_name="Total Stock",
                    y_axis=y_stock,
                    symbol_size=8,
                    is_hover_animation=False,
                    label_opts=opts.LabelOpts(is_show=True, color='blue'),
                    linestyle_opts=opts.LineStyleOpts(width=1.5, color='blue'),
                    is_smooth=True,
                    itemstyle_opts=opts.ItemStyleOpts(color="blue"),
                )
                .set_global_opts(
                            title_opts=opts.TitleOpts(title="Total Stock", pos_top="top", pos_left='left', pos_right='left'),
                            xaxis_opts=opts.AxisOpts(type_="category", name='Date', boundary_gap=False, axisline_opts=opts.AxisLineOpts(is_on_zero=True)),
                            yaxis_opts=opts.AxisOpts(type_="value", is_scale=True, splitline_opts=opts.SplitLineOpts(is_show=True)),
                            legend_opts=opts.LegendOpts(pos_left="center"),
                            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
                            datazoom_opts=[
                                opts.DataZoomOpts(is_realtime=True, type_="inside", xaxis_index=[0, 1], range_start=45, range_end=65),
                                opts.DataZoomOpts(is_realtime=True, type_="slider", xaxis_index=[0, 1], range_start=45, range_end=65, pos_bottom='55px'),
                            ],
                    )
            )
            l2 = (
                Line()
                .add_xaxis(xaxis_data=x)
                .add_yaxis(
                    series_name="Step Profit",
                    y_axis=y_profit,
                    symbol_size=8,
                    is_hover_animation=False,
                    label_opts=opts.LabelOpts(is_show=True, color='orange'),
                    linestyle_opts=opts.LineStyleOpts(width=1.5, color='orange'),
                    is_smooth=True,
                    itemstyle_opts=opts.ItemStyleOpts(color="orange"),
                )
                .add_yaxis(
                    series_name="Cumulative Profit",
                    y_axis=y_profit_cum,
                    symbol_size=8,
                    is_hover_animation=False,
                    label_opts=opts.LabelOpts(is_show=True, color='green'),
                    linestyle_opts=opts.LineStyleOpts(width=1.5, color='green'),
                    is_smooth=True,
                    itemstyle_opts=opts.ItemStyleOpts(color="green"),
                )
                .set_global_opts(
                        title_opts=opts.TitleOpts(
                            title="Profit Details", subtitle="", pos_left="left", pos_right='left', pos_top='45%'
                        ),
                        xaxis_opts=opts.AxisOpts(type_="category", name='Date',
                                                    axisline_opts=opts.AxisLineOpts(is_on_zero=True)),
                        yaxis_opts=opts.AxisOpts(type_="value", is_scale=True,
                                                    splitline_opts=opts.SplitLineOpts(is_show=True),
                                                    ),
                        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
                        legend_opts=opts.LegendOpts(pos_left="center", pos_right="center", pos_top='45%'),
                        datazoom_opts=[
                                opts.DataZoomOpts(is_realtime=True, type_="inside", xaxis_index=[0, 1], range_start=45, range_end=65),
                                opts.DataZoomOpts(is_realtime=True, type_="slider", xaxis_index=[0, 1], range_start=45, range_end=65, pos_bottom='55px'),
                            ],
                    )
                
            )
            grid = (
                        Grid()
                        .add(l1, grid_opts=opts.GridOpts(pos_left=100, pos_right=100, height="30%"))
                        .add(l2, grid_opts=opts.GridOpts(pos_left=100, pos_right=100, pos_top="50%", height="30%"))
                    )
                
            # 时间线轮播多图添加图表实例
            tl.add(grid, "{}".format(self.start_dt.strftime("%Y-%m-%d")))
            tl.render(file_name)

    def render_sku(self):
        df_all = pd.read_csv(self.log_path)
        facility_list = df_all['facility_name'].unique().tolist()
        for facility_name in facility_list:
            df_facility = df_all[df_all['facility_name'] == facility_name]
            sku_list = df_facility['name'].unique().tolist()
            for sku_name in sku_list:
                df = df_facility[df_facility['name'] == sku_name]
                file_name = os.path.join(self.dir_loc, f"{facility_name}_{sku_name}.html")
                if df.shape[0] > 0:
                    df.loc[:, 'sale_dt'] = df['tick'].map(lambda x: self.start_dt+timedelta(days=x))
                    df.sort_values(by='sale_dt', inplace=True)
                    df.loc[:, 'sale_dt'] = df['sale_dt'].map(lambda x: x.strftime('%Y-%m-%d'))
                    x = df['sale_dt'].tolist()
                    y_demand = df['seller_demand'].tolist()
                    y_sales = df['seller_sold'].tolist()
                    df.loc[:, 'inventory_in_stock'] = df['inventory_in_stock'].map(lambda x: np.sum([float(s) for s in x[1:-1].split(",")]))
                    df.loc[:, 'unit_inventory_holding_cost'] = df['unit_inventory_holding_cost'].map(lambda x: np.sum([float(s) for s in x[1:-1].split(",")]))
                    y_stock = [int(x) for x in df['inventory_in_stock'].tolist()]
                    y_consumer = [int(x) for x in df['consumer_purchased'].tolist()]
                    y_received = [int(x) for x in df['consumer_received'].tolist()]
                    y_inventory_holding_cost = [round(x,0) for x in (df['unit_inventory_holding_cost']*df['inventory_in_stock']).tolist()]
                    y_gmv = [round(x,0) for x in (df['seller_price'] * df['seller_sold']).tolist()]
                    y_product_cost = [round(x,0) for x in (df['consumer_order_product_cost']+df['consumer_order_cost']).tolist()]
                    y_oos_loss = [round(x,0) for x in ((df['seller_demand'] - df['seller_sold']) * df['seller_price'] * df['seller_backlog_ratio']).tolist()]
                    y_distribution_cost = [round(x,0) for x in (df['product_transportation_cost'] + df['product_delay_order_penalty']).tolist()]
                else:
                    x = [self.start_dt.strftime('%Y-%m-%d')]
                    y_demand = [0]
                    y_sales = [0]
                    y_stock = [0]
                    y_consumer = [0]
                    y_received = [0]
                    y_inventory_holding_cost = [0]
                    y_gmv = [0]
                    y_product_cost = [0]
                    y_oos_loss = [0]
                    y_distribution_cost = [0]

                tl = Timeline(init_opts=opts.InitOpts(width="1500px", height="800px"))
                tl.add_schema(pos_bottom="bottom", is_auto_play=False, label_opts=opts.LabelOpts(is_show=True, position="bottom"))

                l1 = (
                    Line()
                    .add_xaxis(xaxis_data=x)
                    .add_yaxis(
                        series_name="Sales",
                        y_axis=y_sales,
                        symbol_size=8,
                        is_hover_animation=False,
                        label_opts=opts.LabelOpts(is_show=True, color='blue'),
                        linestyle_opts=opts.LineStyleOpts(width=1.5, color='blue'),
                        is_smooth=True,
                        itemstyle_opts=opts.ItemStyleOpts(color="blue"),
                    )
                    .add_yaxis(
                        series_name="Demand",
                        y_axis=y_demand,
                        symbol_size=8,
                        is_hover_animation=False,
                        label_opts=opts.LabelOpts(is_show=True, color='orange'),
                        linestyle_opts=opts.LineStyleOpts(width=1.5, color='orange'),
                        is_smooth=True,
                        itemstyle_opts=opts.ItemStyleOpts(color="orange"),
                    )
                    .add_yaxis(
                        series_name="Stock",
                        y_axis=y_stock,
                        symbol_size=8,
                        is_hover_animation=False,
                        label_opts=opts.LabelOpts(is_show=True, color='green'),
                        linestyle_opts=opts.LineStyleOpts(width=1.5, color='green'),
                        is_smooth=True,
                        itemstyle_opts=opts.ItemStyleOpts(color="green"),
                    )
                    .add_yaxis(
                        series_name="Replenishing Quantity",
                        y_axis=y_consumer,
                        symbol_size=8,
                        is_hover_animation=False,
                        label_opts=opts.LabelOpts(is_show=True, color='aqua'),
                        linestyle_opts=opts.LineStyleOpts(width=1.5, color='aqua'),
                        is_smooth=True,
                        itemstyle_opts=opts.ItemStyleOpts(color="aqua"),
                    )
                    .add_yaxis(
                        series_name="Inventory Received",
                        y_axis=y_received,
                        symbol_size=8,
                        is_hover_animation=False,
                        label_opts=opts.LabelOpts(is_show=True, color='green'),
                        linestyle_opts=opts.LineStyleOpts(width=1.5, color='green'),
                        is_smooth=True,
                        itemstyle_opts=opts.ItemStyleOpts(color="green"),
                    )
                    .set_global_opts(
                            title_opts=opts.TitleOpts(title="Replenishing Decisions", pos_top="top", pos_left='left', pos_right='left'),
                            xaxis_opts=opts.AxisOpts(type_="category", name='Date', boundary_gap=False, axisline_opts=opts.AxisLineOpts(is_on_zero=True)),
                            yaxis_opts=opts.AxisOpts(type_="value", is_scale=True, splitline_opts=opts.SplitLineOpts(is_show=True)),
                            legend_opts=opts.LegendOpts(pos_left="center"),
                            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
                            datazoom_opts=[
                                opts.DataZoomOpts(is_realtime=True, type_="inside", xaxis_index=[0, 1], range_start=45, range_end=65),
                                opts.DataZoomOpts(is_realtime=True, type_="slider", xaxis_index=[0, 1], range_start=45, range_end=65, pos_bottom='55px'),
                            ],
                    )
                )
                bar =(Bar()
                    .add_xaxis(xaxis_data=x)
                    .add_yaxis(
                        series_name="Holding Cost",
                        y_axis=y_inventory_holding_cost,
                        itemstyle_opts=opts.ItemStyleOpts(color="#123456"),
                        label_opts=opts.LabelOpts(is_show=True),
                    )
                    .add_yaxis(
                        series_name="GMV",
                        y_axis=y_gmv,
                        itemstyle_opts=opts.ItemStyleOpts(color="orange"),
                        label_opts=opts.LabelOpts(is_show=True),
                    )
                    .add_yaxis(
                        series_name="Replenishing Cost",
                        y_axis=y_product_cost,
                        itemstyle_opts=opts.ItemStyleOpts(color="green"),
                        label_opts=opts.LabelOpts(is_show=True),
                    )
                    .add_yaxis(
                        series_name="Out of Stock Cost",
                        y_axis=y_oos_loss,
                        itemstyle_opts=opts.ItemStyleOpts(color="aqua"),
                        label_opts=opts.LabelOpts(is_show=True),
                    )
                    .add_yaxis(
                        series_name="Fulfillment Cost",
                        y_axis=y_distribution_cost,
                        itemstyle_opts=opts.ItemStyleOpts(color="source"),
                        label_opts=opts.LabelOpts(is_show=True),
                    )
                    .set_global_opts(
                        title_opts=opts.TitleOpts(
                            title="Cost Details", subtitle="", pos_left="left", pos_right='left', pos_top='45%'
                        ),
                        xaxis_opts=opts.AxisOpts(type_="category", name='时间',
                                                    axisline_opts=opts.AxisLineOpts(is_on_zero=True)),
                        yaxis_opts=opts.AxisOpts(type_="value", is_scale=True,
                                                    splitline_opts=opts.SplitLineOpts(is_show=True),
                                                    ),
                        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
                        legend_opts=opts.LegendOpts(pos_left="center", pos_right="center", pos_top='45%'),
                        datazoom_opts=[
                                opts.DataZoomOpts(is_realtime=True, type_="inside", xaxis_index=[0, 1], range_start=45, range_end=65),
                                opts.DataZoomOpts(is_realtime=True, type_="slider", xaxis_index=[0, 1], range_start=45, range_end=65, pos_bottom='55px'),
                            ],
                    )
                )

                grid = (
                        Grid()
                        .add(l1, grid_opts=opts.GridOpts(pos_left=100, pos_right=100, height="30%"))
                        .add(bar, grid_opts=opts.GridOpts(pos_left=100, pos_right=100, pos_top="50%", height="30%"))
                    )
                
                # 时间线轮播多图添加图表实例
                tl.add(grid, "{}".format(self.start_dt.strftime("%Y-%m-%d")))
                tl.render(file_name)


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
            sku_name = f"{facility.name}_{sku.name}_{entity.id}"
            facility_type_dict[entity.id] = facility.class_name.__name__
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
            _agent_list.append(f"{facility.name}_{entity_id}")
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

if __name__ == "__main__":
    html_render = SimulationTrackerHtml("/data/songlei/maro_ms/examples/supply_chain/results/dqn_15skus_tr_round1/output_product_metrics_20.csv")
    html_render.render_sku()
    html_render.render_facility()