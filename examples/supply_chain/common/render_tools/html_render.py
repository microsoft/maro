# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pyecharts.options as opts
from pyecharts.charts import Line, Bar, Timeline, Grid


COLORS = [
    "Blue", "Orange", "Green", "CadetBlue", "SteelBlue", "SkyBlue",
    "RoyalBlue", "Aquamarine", "MediumTurquoise", "YellowGreen", "DarkGreen"
]

def compute_store_balance(row):
    return (
        row["seller_sold"] * row['seller_price']
        - (row["seller_demand"] - row['seller_sold']) * row['seller_price'] * row['seller_backlog_ratio']
        - row["consumer_order_product_cost"]
        - row["consumer_order_base_cost"]
        - row['unit_inventory_holding_cost'] * row['inventory_in_stock']
    )

def compute_warehouse_balance(row):
    return (
        row['product_check_in_quantity_in_order'] * row['product_price']
        - row['product_delay_order_penalty']
        - row['product_transportation_cost']
        - row['unit_inventory_holding_cost'] * row['inventory_in_stock']
    )

def compute_supplier_balance(row):
    return (
        row['product_check_in_quantity_in_order'] * row['product_price']
        - row['manufacture_manufacture_cost'] * row['manufacture_finished_quantity']
        - row['product_delay_order_penalty']
        - row['product_transportation_cost']
        - row['unit_inventory_holding_cost'] * row['inventory_in_stock']
    )


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


class SimulationTrackerHtml:
    def __init__(self, log_path, start_dt='2022-01-01'):
        self.log_path = log_path
        self.start_dt = datetime.strptime(start_dt, "%Y-%m-%d")
        self.dir_loc = os.path.dirname(self.log_path)

    def render_facility(self):
        df_all = pd.read_csv(self.log_path)
        facility_list = df_all['facility_name'].unique().tolist()
        for facility_name in facility_list:
            if facility_name.startswith("VNDR"):
                continue
            if facility_name.startswith('supplier'):
                compute_balance = compute_supplier_balance
            elif facility_name.startswith('warehouse'):
                compute_balance = compute_warehouse_balance
            else:
                compute_balance = compute_store_balance
            df = df_all[df_all['facility_name'] == facility_name]
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
            tl.add_schema(
                pos_bottom="bottom", is_auto_play=False, label_opts=opts.LabelOpts(is_show=True, position="bottom")
            )
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
                    xaxis_opts=opts.AxisOpts(
                        type_="category", name='Date', boundary_gap=False,
                        axisline_opts=opts.AxisLineOpts(is_on_zero=True)
                    ),
                    yaxis_opts=opts.AxisOpts(
                        type_="value", is_scale=True, splitline_opts=opts.SplitLineOpts(is_show=True)
                    ),
                    legend_opts=opts.LegendOpts(pos_left="center"),
                    tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
                    datazoom_opts=[
                        opts.DataZoomOpts(
                            is_realtime=True, type_="inside", xaxis_index=[0, 1], range_start=45, range_end=65
                        ),
                        opts.DataZoomOpts(
                            is_realtime=True, type_="slider", xaxis_index=[0, 1], range_start=45, range_end=65,
                            pos_bottom='55px'
                        ),
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
                    xaxis_opts=opts.AxisOpts(
                        type_="category", name='Date', axisline_opts=opts.AxisLineOpts(is_on_zero=True)
                    ),
                    yaxis_opts=opts.AxisOpts(
                        type_="value", is_scale=True, splitline_opts=opts.SplitLineOpts(is_show=True),
                    ),
                    tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
                    legend_opts=opts.LegendOpts(pos_left="center", pos_right="center", pos_top='45%'),
                    datazoom_opts=[
                        opts.DataZoomOpts(
                            is_realtime=True, type_="inside", xaxis_index=[0, 1], range_start=45, range_end=65
                        ),
                        opts.DataZoomOpts(
                            is_realtime=True, type_="slider", xaxis_index=[0, 1], range_start=45, range_end=65,
                            pos_bottom='55px'
                        ),
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
            if facility_name.startswith("VNDR"):
                continue
            df_facility = df_all[df_all['facility_name'] == facility_name]
            sku_list = df_facility['name'].unique().tolist()
            for sku_name in sku_list:
                df = df_facility[df_facility['name'] == sku_name]
                file_name = os.path.join(self.dir_loc, f"{facility_name}_{sku_name}.html")
                if df.shape[0] > 0:
                    df.loc[:, 'sale_dt'] = df['tick'].map(lambda x: self.start_dt + timedelta(days=x))
                    df.sort_values(by='sale_dt', inplace=True)
                    df.loc[:, 'sale_dt'] = df['sale_dt'].map(lambda x: x.strftime('%Y-%m-%d'))
                    x = df['sale_dt'].tolist()
                    y_demand = df['seller_demand'].tolist()
                    y_sales = df['seller_sold'].tolist()
                    y_stock = [int(x) for x in df['inventory_in_stock'].tolist()]
                    y_consumer = [int(x) for x in df['consumer_purchased'].tolist()]
                    y_received = [int(x) for x in df['consumer_received'].tolist()]
                    y_inventory_holding_cost = [
                        round(x, 0) for x in (df['unit_inventory_holding_cost'] * df['inventory_in_stock']).tolist()
                    ]
                    y_gmv = [
                        round(x, 0) for x in (
                            df['seller_price'] * df['seller_sold']
                            + df['product_check_in_quantity_in_order'] * df['product_price']
                        ).tolist()
                    ]
                    y_product_cost = [
                        round(x, 0) for x in (df['consumer_order_product_cost']+df['consumer_order_base_cost']).tolist()
                    ]
                    y_oos_loss = [
                        round(x, 0) for x in (
                            (df['seller_demand'] - df['seller_sold']) * df['seller_price'] * df['seller_backlog_ratio']
                        ).tolist()
                    ]
                    y_distribution_cost = [
                        round(x, 0) for x in (
                            df['product_transportation_cost'] + df['product_delay_order_penalty']
                        ).tolist()
                    ]
                    y_manufacture_quantity = [round(x, 0) for x in df['manufacture_finished_quantity']]
                    y_manufacture_cost = [
                        round(x, 0) for x in df['manufacture_finished_quantity'] * df['manufacture_manufacture_cost']
                    ]
                    y_in_transit = [round(x, 0) for x in df['inventory_in_transit']]
                    y_to_distribute = [round(x, 0) for x in df['inventory_to_distribute']]
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
                    y_manufacture_quantity = [0]
                    y_manufacture_cost = [0]
                    y_in_transit = [0]
                    y_to_distribute = [0]

                tl = Timeline(init_opts=opts.InitOpts(width="1500px", height="800px"))
                tl.add_schema(
                    pos_bottom="bottom", is_auto_play=False, label_opts=opts.LabelOpts(is_show=True, position="bottom")
                )

                l1 = Line().add_xaxis(xaxis_data=x)

                l1_to_render = [
                    ("Sales", y_sales),
                    ("Demand", y_demand),
                    ("Stock", y_stock),
                    ("Replenishing Quantity", y_consumer),
                    ("Inventory Received", y_received),
                    ("Manufacture Quantity", y_manufacture_quantity),
                    ("Products in Transit", y_in_transit),
                    ("Products to Distribute", y_to_distribute),
                ]
                for i, (l_name, vals) in enumerate(l1_to_render):
                    if np.all(np.array(vals) == 0):
                        continue
                    l1.add_yaxis(
                        series_name=l_name,
                        y_axis=vals,
                        symbol_size=8,
                        is_hover_animation=False,
                        label_opts=opts.LabelOpts(is_show=True, color=COLORS[i]),
                        linestyle_opts=opts.LineStyleOpts(width=1.5, color=COLORS[i]),
                        is_smooth=True,
                        itemstyle_opts=opts.ItemStyleOpts(color=COLORS[i]),
                    )
                l1.set_global_opts(
                    title_opts=opts.TitleOpts(
                        title="Replenishing Decisions", pos_top="top", pos_left='left', pos_right='left'
                    ),
                    xaxis_opts=opts.AxisOpts(
                        type_="category", name='Date', boundary_gap=False,
                        axisline_opts=opts.AxisLineOpts(is_on_zero=True)
                    ),
                    yaxis_opts=opts.AxisOpts(
                        type_="value", is_scale=True, splitline_opts=opts.SplitLineOpts(is_show=True)
                    ),
                    legend_opts=opts.LegendOpts(pos_left="center"),
                    tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
                    datazoom_opts=[
                        opts.DataZoomOpts(
                            is_realtime=True, type_="inside", xaxis_index=[0, 1], range_start=45, range_end=65
                        ),
                        opts.DataZoomOpts(
                            is_realtime=True, type_="slider", xaxis_index=[0, 1], range_start=45, range_end=65,
                            pos_bottom='55px'
                        ),
                    ],
                )

                bar_to_render = [
                    ("Holding Cost", y_inventory_holding_cost),
                    ("GMV", y_gmv),
                    ("Replenishing Cost", y_product_cost),
                    ("Out of Stock Cost", y_oos_loss),
                    ("Fulfillment Cost", y_distribution_cost),
                    ("Manufacture Cost", y_manufacture_cost)
                ]

                bar = Bar().add_xaxis(xaxis_data=x)

                for i, (l_name, vals) in enumerate(bar_to_render):
                    if np.all(np.array(vals) == 0):
                        continue
                    bar.add_yaxis(
                        series_name=l_name,
                        y_axis=vals,
                        itemstyle_opts=opts.ItemStyleOpts(color=COLORS[i]),
                        label_opts=opts.LabelOpts(is_show=True),
                    )

                bar.set_global_opts(
                    title_opts=opts.TitleOpts(
                        title="Cost Details", subtitle="", pos_left="left", pos_right='left', pos_top='45%'
                    ),
                    xaxis_opts=opts.AxisOpts(
                        type_="category", name='Time', axisline_opts=opts.AxisLineOpts(is_on_zero=True)
                    ),
                    yaxis_opts=opts.AxisOpts(
                        type_="value", is_scale=True, splitline_opts=opts.SplitLineOpts(is_show=True),
                    ),
                    tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
                    legend_opts=opts.LegendOpts(pos_left="center", pos_right="center", pos_top='45%'),
                    datazoom_opts=[
                        opts.DataZoomOpts(
                            is_realtime=True, type_="inside", xaxis_index=[0, 1], range_start=45, range_end=65
                        ),
                        opts.DataZoomOpts(
                            is_realtime=True, type_="slider", xaxis_index=[0, 1], range_start=45, range_end=65,
                            pos_bottom='55px'
                        ),
                    ],
                )

                grid = (
                    Grid()
                    .add(l1, grid_opts=opts.GridOpts(pos_left=100, pos_right=100, height="30%"))
                    .add(bar, grid_opts=opts.GridOpts(pos_left=100, pos_right=100, pos_top="50%", height="30%"))
                )

                # 时间线轮播多图添加图表实例
                tl.add(grid, "{}".format(self.start_dt.strftime("%Y-%m-%d")))
                tl.render(file_name)
