# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pyecharts.options as opts
from pyecharts.charts import Bar, Grid, Tab, PictorialBar
from pyecharts.components import Table
from pyecharts.options import ComponentTitleOpts
from pyecharts.globals import SymbolType


"""
tick,id,sku_id,facility_id,
facility_name,name,inventory_in_stock,
inventory_in_transit, inventory_to_distribute,
unit_inventory_holding_cost,
consumer_purchased,
consumer_received,consumer_order_base_cost,
consumer_order_product_cost,seller_sold,seller_demand,
seller_price,seller_backlog_ratio,
manufacture_finished_quantity,
manufacture_manufacture_cost,product_price,
product_check_in_quantity_in_order,product_delay_order_penalty,
product_transportation_cost
distribution_pending_product_quantity,
distribution_pending_order_number
"""


class SimulationComparisonTrackerHtml:
    def __init__(self, model1, log_path1, model2, log_path2, start_dt='2022-01-01'):
        self.log_path1 = log_path1
        self.model1 = model1
        self.log_path2 = log_path2
        self.model2 = model2
        self.start_dt = datetime.strptime(start_dt, "%Y-%m-%d")
        self.dir_loc = os.path.dirname(self.log_path2)
        self.df = self._load_data()

    def _load_data(self):
        df1 = pd.read_csv(self.log_path1)
        df1.loc[:, "model_name"] = self.model1
        df2 = pd.read_csv(self.log_path2)
        df2.loc[:, "model_name"] = self.model2
        df = pd.concat([df1, df2])
        df.loc[:, 'sale_dt'] = df['tick'].map(lambda x: self.start_dt+timedelta(days=x))
        df = df[(df['sale_dt'] >= self.start_dt+timedelta(days=180))]
        self.facility_name_list = [
            facility_name for facility_name in df['facility_name'].unique()
            if len(facility_name) > 2 and facility_name[:2] in ['CA', 'TX', 'WI']
        ]
        df = df[df['facility_name'].isin(self.facility_name_list)]
        return df

    def render_overview(self):
        df_agg = self.df.copy(deep=True)
        df_agg = df_agg.groupby(['facility_name', 'sku_id', 'sale_dt', 'model_name']).first().reset_index()
        df_agg.loc[:, "facility_name"] = 'ALL'
        df_agg.loc[:, "facility_id"] = -1
        df = self.df.groupby(['facility_name', 'sku_id', 'sale_dt', 'model_name']).first().reset_index()
        df_sku = pd.concat([df, df_agg])

        df_sku.loc[:, "GMV"] = df_sku['seller_price'] * df_sku['seller_sold']
        df_sku.loc[:, "order_cost"] = df_sku["consumer_order_product_cost"] + df_sku["consumer_order_base_cost"]
        df_sku.loc[:, 'inventory_holding_cost'] = df_sku['inventory_in_stock'] * df_sku['unit_inventory_holding_cost']
        df_sku.loc[:, 'out_of_stock_loss'] = (
            df_sku['seller_backlog_ratio'] * df_sku['seller_price'] * (df_sku['seller_demand'] - df_sku['seller_sold'])
        )
        df_sku.loc[:, "profit"] = (
            df_sku["GMV"] - df_sku['order_cost'] - df_sku['inventory_holding_cost'] - df_sku['out_of_stock_loss']
        )
        num_days = df_sku['sale_dt'].unique().shape[0]
        cols = [
            'facility_name', 'model_name', 'GMV', 'profit', 'order_cost', 'inventory_in_stock',
            'inventory_holding_cost', 'seller_sold', 'seller_demand'
        ]
        df_sku = df_sku[['name'] + cols].groupby(['facility_name', 'name', 'model_name']).sum().reset_index()
        df_sku.loc[:, "turnover_rate"] =  df_sku['seller_demand'] * num_days / df_sku['inventory_in_stock']
        df_sku.loc[:, "available_rate"] = df_sku['seller_sold'] / df_sku['seller_demand']
        cols.extend(['turnover_rate', 'available_rate'])
        agg_func = {
            col: np.mean if col in ['turnover_rate', 'available_rate'] else np.sum
            for col in cols[2:]
        }

        df = (
            df_sku[cols]
            .groupby(['facility_name', 'model_name'])
            .agg(agg_func)
            .reset_index()
        )
        df_sku.sort_values(by=['name', 'facility_name', 'model_name'], inplace=True)

        details_headers = ['name'] + cols
        details_rows = df_sku[details_headers].values.tolist()


        df.loc[:, 'x'] = df.apply(lambda x: f"{x['facility_name']}_{x['model_name']}", axis=1)
        x = df['x'].tolist()
        y_gmv = [round(x,2) for x in df['GMV'].tolist()]
        y_profit = [round(x,2) for x in df['profit'].tolist()]
        y_order_cost = [round(x,2) for x in df['order_cost'].tolist()]
        y_inventory_holding_cost = [round(x,2) for x in df['inventory_holding_cost'].tolist()]
        y_seller_sold = [round(x,2) for x in df['seller_sold'].tolist()]
        y_turnover_rate = [round(x,3) for x in df['turnover_rate'].tolist()]
        y_available_rate = [round(x,3) for x in df['available_rate'].tolist()]

        tab = Tab()

        for (name, y_vals) in zip(
            [
                'GMV (¥)', 'Profit (¥)', 'Inventory Holding Cost (¥)', 'Order Cost (¥)',
                'Total Sales (Units)', 'Turnover Rate (Days)', 'Available Rate'
            ], [
                y_gmv, y_profit, y_inventory_holding_cost, y_order_cost,
                y_seller_sold, y_turnover_rate, y_available_rate
            ]
        ):
            c = (
                PictorialBar(opts.InitOpts(height=f'{100*len(self.facility_name_list)}px', width='1200px'))
                .add_xaxis(x)
                .add_yaxis(
                    "",
                    y_vals,
                    label_opts=opts.LabelOpts(is_show=True, position="right"),
                    symbol_size=15,
                    symbol_repeat="fixed",
                    symbol_offset=[0, 0],
                    is_symbol_clip=True,
                    symbol=SymbolType.ROUND_RECT,
                )
                .reversal_axis()
                .set_global_opts(
                    title_opts=opts.TitleOpts(title=""),
                    xaxis_opts=opts.AxisOpts(is_show=False),
                    legend_opts=opts.LegendOpts(pos_left="center", pos_right="center", pos_top='45%'),
                    yaxis_opts=opts.AxisOpts(
                        axistick_opts=opts.AxisTickOpts(is_show=False),
                        axisline_opts=opts.AxisLineOpts(
                            linestyle_opts=opts.LineStyleOpts(opacity=0)
                        ),
                    ),
                    datazoom_opts=opts.DataZoomOpts(orient='vertical'),
                )
            )

            model1_larger_cnt, model2_larger_cnt = 0, 0
            for facility in self.facility_name_list:
                idx1 = x.index(f"{facility}_{self.model1}")
                idx2 = x.index(f"{facility}_{self.model2}")
                if y_vals[idx2] > y_vals[idx1]: model2_larger_cnt += 1
                else: model1_larger_cnt += 1

            b=(
                Bar()
                .add_xaxis([f'{self.model1}>{self.model2}', f'{self.model2}>{self.model1}'])
                .add_yaxis(
                    f'larger {name} counts',
                    [int(model1_larger_cnt), int(model2_larger_cnt)],
                    color='RoyalBlue',
                    bar_width="20px",
                )
                .set_global_opts(
                    title_opts=opts.TitleOpts(title=name),
                    yaxis_opts=opts.AxisOpts(name="Number of Facilities"),
                    xaxis_opts=opts.AxisOpts(name="Models"),
                    legend_opts=opts.LegendOpts(pos_left="center", pos_right="center", pos_top='70%'),
                )
            )

            g = (
                Grid(opts.InitOpts(height='1200px', width='1200px'))
                .add(c, opts.GridOpts(pos_left=100, pos_right=100, height="60%"), is_control_axis_index=True)
                .add(b, opts.GridOpts(pos_top="70%", height="25%"))
            )
            tab.add(g, name)

        table = Table()
        table.add(details_headers, details_rows)
        table.set_global_opts(
            title_opts=ComponentTitleOpts(title="SKU Cost Items", subtitle="")
        )

        tab.add(table, "Details")
        tab.render(os.path.join(self.dir_loc, "comparision_overview.html"))
