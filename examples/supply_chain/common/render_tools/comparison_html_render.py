# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from datetime import datetime, timedelta
from typing import Counter, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyecharts.options as opts
from pyecharts.charts import Bar, Grid, PictorialBar, Tab
from pyecharts.components import Table
from pyecharts.globals import SymbolType
from pyecharts.options import ComponentTitleOpts

"""
tick, id, facility_id, facility_name, sku_id, name, product_price,
unit_inventory_holding_cost, inventory_in_stock, inventory_in_transit, inventory_to_distribute,
product_check_in_quantity_in_order, product_delay_order_penalty, product_transportation_cost,
consumer_purchased, consumer_received, consumer_order_base_cost, consumer_order_product_cost,
seller_sold, seller_demand, seller_backlog_ratio,
manufacture_finished_quantity, manufacture_in_pipeline_quantity,
manufacture_manufacture_cost, manufacture_start_manufacture_quantity,
distribution_pending_product_quantity, distribution_pending_order_number,
"""


class SimulationComparisonTrackerHtml:
    def __init__(
        self,
        name_path_list: List[Tuple[str, str]],
        dump_dir: str,
        log_dir: str,
        dump_name: Optional[str] = None,
        csv_name: str = "output_product_metrics.csv",
        start_date: str = "2022-01-01",
        train_days: int = 180,
    ):
        self.exp_name_list = [name for (name, _) in name_path_list]
        self.exp_path_list = [os.path.join(log_dir, name, csv_name) for (_, name) in name_path_list]

        os.makedirs(dump_dir, exist_ok=True)
        if dump_name is None:
            dump_name = "_".join(self.exp_name_list)
        self.dump_path = os.path.join(dump_dir, f"{dump_name}_comparison.html")

        self.start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        self.train_days = train_days

        self.df = self._load_data()

    def _load_data(self):
        df_list = []
        for name, csv_path in zip(self.exp_name_list, self.exp_path_list):
            df_i = pd.read_csv(csv_path)
            df_i.loc[:, "Exp Name"] = name
            df_list.append(df_i)
        df = pd.concat(df_list)

        df.loc[:, "sale_dt"] = df["tick"].map(lambda x: self.start_dt + timedelta(days=x))
        df = df[(df["sale_dt"] >= self.start_dt + timedelta(days=self.train_days))]
        self.facility_name_list = [
            facility_name
            for facility_name in df["facility_name"].unique()
            if len(facility_name) > 2 and facility_name[:2] in ["CA", "TX", "WI"]
        ]
        df = df[df["facility_name"].isin(self.facility_name_list)]
        return df

    def render_overview(self):
        df_agg = self.df.copy(deep=True)
        df_agg = df_agg.groupby(["facility_name", "sku_id", "sale_dt", "Exp Name"]).first().reset_index()
        df_agg.loc[:, "facility_name"] = "ALL"
        df_agg.loc[:, "facility_id"] = -1
        df = self.df.groupby(["facility_name", "sku_id", "sale_dt", "Exp Name"]).first().reset_index()
        df_sku = pd.concat([df, df_agg])

        df_sku.loc[:, "GMV"] = df_sku["product_price"] * df_sku["seller_sold"]
        df_sku.loc[:, "order_cost"] = df_sku["consumer_order_product_cost"] + df_sku["consumer_order_base_cost"]
        df_sku.loc[:, "inventory_holding_cost"] = df_sku["inventory_in_stock"] * df_sku["unit_inventory_holding_cost"]
        df_sku.loc[:, "out_of_stock_loss"] = (
            df_sku["seller_backlog_ratio"] * df_sku["product_price"] * (df_sku["seller_demand"] - df_sku["seller_sold"])
        )
        df_sku.loc[:, "profit"] = (
            df_sku["GMV"]
            - df_sku["order_cost"]
            - df_sku["inventory_holding_cost"]
            - df_sku["out_of_stock_loss"]
            + df_sku["product_check_in_quantity_in_order"] * df_sku["product_price"]
        )
        num_days = df_sku["sale_dt"].unique().shape[0]
        cols = [
            "facility_name",
            "Exp Name",
            "GMV",
            "profit",
            "order_cost",
            "inventory_in_stock",
            "inventory_holding_cost",
            "seller_sold",
            "seller_demand",
            "product_price",
        ]
        df_sku = df_sku[["name"] + cols].groupby(["facility_name", "name", "Exp Name"]).sum().reset_index()
        df_sku.loc[:, "turnover_rate"] = df_sku["seller_demand"] * num_days / df_sku["inventory_in_stock"]
        df_sku.loc[:, "available_rate"] = df_sku["seller_sold"] / df_sku["seller_demand"]
        cols.extend(["turnover_rate", "available_rate"])
        agg_func = {col: np.mean if col in ["turnover_rate", "available_rate"] else np.sum for col in cols[2:]}

        df = df_sku[cols].groupby(["facility_name", "Exp Name"]).agg(agg_func).reset_index()
        df_sku.sort_values(by=["name", "facility_name"], inplace=True)

        details_headers = ["name"] + cols
        details_rows = df_sku[details_headers].values.tolist()

        header_mapping = {
            "facility_name": "Facility",
            "product_price": "Price",
        }
        details_headers = [header_mapping.get(header, header) for header in details_headers]

        df.loc[:, "x"] = df.apply(lambda x: f"{x['facility_name']}_{x['Exp Name']}", axis=1)
        x = df["x"].tolist()
        y_gmv = [round(x, 2) for x in df["GMV"].tolist()]
        y_profit = [round(x, 2) for x in df["profit"].tolist()]
        y_order_cost = [round(x, 2) for x in df["order_cost"].tolist()]
        y_inventory_holding_cost = [round(x, 2) for x in df["inventory_holding_cost"].tolist()]
        y_seller_sold = [round(x, 2) for x in df["seller_sold"].tolist()]
        y_turnover_rate = [round(x, 3) for x in df["turnover_rate"].tolist()]
        y_available_rate = [round(x, 3) for x in df["available_rate"].tolist()]

        tab = Tab()

        for (name, y_vals) in zip(
            [
                "GMV (¥)",
                "Profit (¥)",
                "Inventory Holding Cost (¥)",
                "Order Cost (¥)",
                "Total Sales (Units)",
                "Turnover Rate (Days)",
                "Available Rate",
            ],
            [
                y_gmv,
                y_profit,
                y_inventory_holding_cost,
                y_order_cost,
                y_seller_sold,
                y_turnover_rate,
                y_available_rate,
            ],
        ):
            c = (
                PictorialBar(opts.InitOpts(height=f"{100 * len(self.facility_name_list)}px", width="1200px"))
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
                    legend_opts=opts.LegendOpts(pos_left="center", pos_right="center", pos_top="45%"),
                    yaxis_opts=opts.AxisOpts(
                        axistick_opts=opts.AxisTickOpts(is_show=False),
                        axisline_opts=opts.AxisLineOpts(
                            linestyle_opts=opts.LineStyleOpts(opacity=0),
                        ),
                    ),
                    datazoom_opts=opts.DataZoomOpts(orient="vertical"),
                )
            )

            best_count = Counter()
            for facility in self.facility_name_list:
                best_name, best_val = self.exp_name_list[0], y_vals[x.index(f"{facility}_{self.exp_name_list[0]}")]
                for exp_name in self.exp_name_list:
                    idx = x.index(f"{facility}_{exp_name}")
                    val = y_vals[idx]
                    if val > best_val:
                        best_name = exp_name
                        best_val = val
                best_count[best_name] += 1

            b = (
                Bar()
                .add_xaxis([f"{name} best" for name in self.exp_name_list])
                .add_yaxis(
                    f"larger {name} counts",
                    [best_count[name] for name in self.exp_name_list],
                    color="RoyalBlue",
                    bar_width="20px",
                )
                .set_global_opts(
                    title_opts=opts.TitleOpts(title=name),
                    yaxis_opts=opts.AxisOpts(name="Number of Facilities"),
                    xaxis_opts=opts.AxisOpts(name="Models"),
                    legend_opts=opts.LegendOpts(is_show=False, pos_bottom="25%", pos_right="10%"),
                )
            )

            g = (
                Grid(opts.InitOpts(height="1200px", width="1200px"))
                .add(c, opts.GridOpts(pos_left=100, pos_right=100, height="60%"), is_control_axis_index=True)
                .add(b, opts.GridOpts(pos_top="70%", height="25%"))
            )
            tab.add(g, name)

        table = Table()
        table.add(details_headers, details_rows)
        table.set_global_opts(
            title_opts=ComponentTitleOpts(title="SKU Cost Items", subtitle=""),
        )

        tab.add(table, "Details")

        tab.render(self.dump_path)


if __name__ == "__main__":
    # Default v.s. Cheapest v.s. Shortest
    comparison_render = SimulationComparisonTrackerHtml(
        name_path_list=[
            ("Default", "SCI_500_default_storage_enlarged_EOQ_vlt-2-2-2"),
            ("Cheapest", "SCI_500_cheapest_storage_enlarged_EOQ_vlt-2-2-2"),
            ("Shortest", "SCI_500_shortest_storage_enlarged_EOQ_vlt-2-2-2"),
        ],
        dump_dir="/home/jinyu/maro/examples/supply_chain/logs/SCI_500_comparison",
        log_dir="/home/jinyu/maro/examples/supply_chain/logs",
        dump_name=None,
    )
    comparison_render.render_overview()

    # VLT Buffer Factor 2x v.s. 1x
    comparison_render = SimulationComparisonTrackerHtml(
        name_path_list=[
            ("2x", "SCI_500_shortest_storage_enlarged_EOQ_vlt-2-2-2"),
            ("1x", "SCI_500_shortest_storage_enlarged_EOQ_vlt-1-1-1"),
        ],
        dump_dir="/home/jinyu/maro/examples/supply_chain/logs/SCI_500_comparison",
        log_dir="/home/jinyu/maro/examples/supply_chain/logs",
        dump_name="Factor_2x_1x",
    )
    comparison_render.render_overview()

    # Storage Capacity 10x v.s. 1x
    comparison_render = SimulationComparisonTrackerHtml(
        name_path_list=[
            ("10x", "SCI_500_shortest_storage_enlarged_EOQ_vlt-1-1-1"),
            ("1x", "SCI_500_shortest_EOQ_vlt-1-1-1"),
        ],
        dump_dir="/home/jinyu/maro/examples/supply_chain/logs/SCI_500_comparison",
        log_dir="/home/jinyu/maro/examples/supply_chain/logs",
        dump_name="Storage_10x_1x",
    )
    comparison_render.render_overview()
