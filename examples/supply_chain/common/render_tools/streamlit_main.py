# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
import pickle
from typing import Dict, List, Tuple

import graphviz
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from maro.simulator.scenarios.supply_chain.facilities import OuterRetailerFacility


# "wide": use the entire screen; "centered": centered into a fixed width.
st.set_page_config(layout="wide")


MARKDOWN_BODY = f"""
    <style>
        .reportview-container .main .block-container{{
            max-width: 3000px;
            margin-left: 15px;
            margin-right: 15px;
        }}

    </style>
"""

LINE_TYPE = {
    "Exp_1": dict(color="deepskyblue", width=1,),
    "Exp_2": dict(color="magenta", width=1,),
    "Exp_3": dict(color="yellowgreen", width=1,),
    "Exp_4": dict(color="orangered", width=1,),
    "Fea_1": dict(color="dodgerblue", width=1,),
    "Fea_2": dict(color="darkorange", width=1,),
    "Fea_3": dict(color="lightseagreen", width=1,),
    "Fea_4": dict(color="violet", width=1,),
}

SHOW_NAME = {
    "deepskyblue": "Blue",
    "magenta": "Magenta",
    "yellowgreen": "Green",
    "orangered": "Orange",
}


def set_exp_option(exp_idx: int, exp_list: List[str], log_dir: str):
    color = SHOW_NAME[LINE_TYPE[f"Exp_{exp_idx}"].__getitem__('color')]
    exp_name = st.sidebar.selectbox(f"Experiment {exp_idx} ({color})", exp_list)
    if exp_name == "DUMMY":
        return None, None, None

    line_col = f"Exp_{exp_idx}"
    exp_log_dir = os.path.join(log_dir, exp_name)
    return exp_name, line_col, exp_log_dir


def _parse_entity_info(sku_status):
    entity_infos = sku_status["entity_infos"]
    len_period = sku_status["balance_status"].shape[1]
    by_fname_and_sku_name = {}
    for (i, id, f_name, sku_name, class_name) in entity_infos:
        prefix = sku_name.split("_")[0]
        if f_name not in by_fname_and_sku_name:
            by_fname_and_sku_name[f_name] = {}
        if prefix not in by_fname_and_sku_name[f_name]:
            by_fname_and_sku_name[f_name][prefix] = {}
        by_fname_and_sku_name[f_name][prefix][sku_name] = (i, id, class_name)
    return by_fname_and_sku_name, len_period


def _parse_facility_info(sku_status):
    facility_infos = sku_status["facility_infos"]
    facility_by_name = {
        f_name: (i, f_id, f_class) for (i, f_id, f_name, f_class) in facility_infos
    }
    return facility_by_name


def _parse_vendor_dict(vendor_dict):
    upstream_set_dict = {
        f_down: set([f_up for f_up, _ in by_sku.values() if not f_up.startswith("VNDR")])
        for f_down, by_sku in vendor_dict.items()
    }
    return upstream_set_dict


def set_exp_list(exp_num: int, exp_list: List[str], log_dir: str):
    exp_name_list, line_col_list, exp_log_dir_list = [], [], []

    for exp_idx in range(exp_num):
        exp_name, line_col, exp_log_dir = set_exp_option(exp_idx + 1, exp_list, log_dir)

        if exp_name is not None:
            exp_name_list.append(exp_name)
            line_col_list.append(line_col)
            exp_log_dir_list.append(exp_log_dir)

    sku_status_list = read_all_data(exp_log_dir_list)

    if len(sku_status_list) > 0:
        by_fname_type_sku, len_period = _parse_entity_info(sku_status_list[0])
        facility_by_name = _parse_facility_info(sku_status_list[0])
    else:
        by_fname_type_sku, len_period = {}, 0
        facility_by_name = {}

    exp_data_list: List[Tuple[str, int, dict]] = [
        (exp_name, line_col, sku_status)
        for exp_name, line_col, sku_status in zip(exp_name_list, line_col_list, sku_status_list)
    ]

    return exp_data_list, by_fname_type_sku, facility_by_name, len_period


def _calculate_metrics(df: pd.DataFrame, len_period: int):
    max_tick = df['tick'].max()
    min_tick = max_tick - len_period
    df = df[min_tick < df['tick']][df['tick'] <= max_tick]

    df.loc[:, "GMV"] = df['product_price'] * df['seller_sold']
    df.loc[:, "order_cost"] = df["consumer_order_product_cost"] + df["consumer_order_base_cost"]
    df.loc[:, 'inventory_holding_cost'] = df['inventory_in_stock'] * df['unit_inventory_holding_cost']
    df.loc[:, 'out_of_stock_loss'] = (
        df['seller_backlog_ratio'] * df['product_price'] * (df['seller_demand'] - df['seller_sold'])
    )
    df.loc[:, "profit"] = (
        df["GMV"] - df['order_cost'] - df['inventory_holding_cost'] - df['out_of_stock_loss']
        + df["product_check_in_quantity_in_order"] * df["product_price"]
    )

    cols = [
        'facility_name', 'GMV', 'profit', 'order_cost', 'inventory_in_stock',
        'inventory_holding_cost', 'seller_sold', 'seller_demand', 'product_price'
    ]
    df = df[['name'] + cols].groupby(['facility_name', 'name']).sum().reset_index()
    df.loc[:, "turnover_rate"] = df['seller_demand'] * len_period / df['inventory_in_stock']
    df.loc[:, "available_rate"] = df['seller_sold'] / df['seller_demand']
    cols.extend(['turnover_rate', 'available_rate'])
    agg_func = {
        col: np.mean if col in ['turnover_rate', 'available_rate'] else np.sum
        for col in cols[2:]
    }
    df = df[cols].groupby(['facility_name']).agg(agg_func).reset_index()

    turnover_rate_dict, available_rate_dict = {}, {}
    for facility, turnover_rate, available_rate in zip(
        df['facility_name'].tolist(), df['turnover_rate'].tolist(), df['available_rate'].tolist()
    ):
        turnover_rate_dict[facility] = round(turnover_rate, 3)
        available_rate_dict[facility] = round(available_rate, 3)

    return turnover_rate_dict, available_rate_dict


@st.cache
def read_data(exp_log_dir: str) -> dict:
    sku_status_path = os.path.join(exp_log_dir, "sku_status.pkl")
    with open(sku_status_path, 'rb') as f:
        sku_status = pickle.load(f)

    network_path = os.path.join(exp_log_dir, "vendor.py")
    with open(network_path, 'r') as f:
        vendor_dict = json.load(f)

    csv_path = os.path.join(exp_log_dir, "output_product_metrics.csv")
    df = pd.read_csv(csv_path)
    len_period = sku_status["balance_status"].shape[1]
    turnover_rate_dict, available_rate_dict = _calculate_metrics(df, len_period)

    sku_status["vendor_dict"] = vendor_dict
    sku_status["upstream_set_dict"] = _parse_vendor_dict(vendor_dict)
    sku_status["turnover_rate_dict"] = turnover_rate_dict
    sku_status["available_rate_dict"] = available_rate_dict

    return sku_status


@st.cache
def read_all_data(exp_log_dir_list: List[str]):
    sku_status_list = []
    for exp_log_dir in exp_log_dir_list:
        sku_status = read_data(exp_log_dir)
        sku_status_list.append(sku_status)
    return sku_status_list


def plot_team_balance(facility_by_name: Dict[str, tuple], len_period: int, exp_data_list: List[Tuple[str, int, dict]]):
    facility_idx_list = [info[0] for info in facility_by_name.values() if issubclass(info[2], OuterRetailerFacility)]

    fig = go.Figure()
    table_data = {"Exp Name": [], "Acc Balance": []}
    for exp_name, line_col, sku_status in exp_data_list:
        balance = sku_status["step_balance"]
        chosen_balance = [balance[0, :, i] for i in facility_idx_list]
        accumulated_balance = np.cumsum(np.sum(chosen_balance, axis=0))

        fig.add_trace(
            go.Scatter(
                x=list(range(len_period)),
                y=accumulated_balance,
                mode="lines",
                legendgroup=exp_name,
                name=exp_name,
                line=LINE_TYPE[line_col],
                showlegend=False,
            ),
        )

        table_data["Exp Name"].append(exp_name)
        table_data["Acc Balance"].append(f"{accumulated_balance[-1]:,.0f}")

    st.plotly_chart(fig, use_container_width=True)

    # st.table(table_data)

    fig = go.Figure(data=go.Table(
        header=dict(
            values=["Exp Name", "Acc Balance"],
            font=dict(size=24),
            height=40,
        ),
        columnwidth=[250, 80],
        cells=dict(
            values=[table_data["Exp Name"], table_data["Acc Balance"]],
            font=dict(size=24),
            height=40,
            align=['left', 'right']
        ),
    ))

    st.plotly_chart(fig)


def plot_facility_balance(f_name: str, idx: int, len_period: int, exp_data_list: List[Tuple[str, int, dict]]):
    fig = go.Figure()
    fig.update_layout(title_text=f"Facility {f_name}")
    for exp_name, line_col, sku_status in exp_data_list:
        balance = sku_status["step_balance"][0, :, idx]

        fig.add_trace(
            go.Scatter(
                x=list(range(len_period)),
                y=np.cumsum(balance),
                mode="lines",
                legendgroup=exp_name,
                name=exp_name,
                line=LINE_TYPE[line_col],
                showlegend=False,
            ),
        )
    st.plotly_chart(fig, use_container_width=True)


def plot_single_sku_status(
    title: str, status_key: str, status_idx: int, len_period: int, exp_data_list: List[Tuple[str, int, dict]]
):
    fig = go.Figure()
    fig.update_layout(title_text=title)
    for exp_name, line_col, sku_status in exp_data_list:
        status = sku_status[status_key][0, :, status_idx]
        fig.add_trace(
            go.Scatter(
                x=list(range(len_period)),
                y=status,
                mode="lines",
                legendgroup=exp_name,
                name=exp_name,
                line=LINE_TYPE[line_col],
                showlegend=False,
            ),
        )
    st.plotly_chart(fig, use_container_width=True)


def plot_stock_related_status(fname: str, status_idx: int, len_period: int, exp_data_list: List[Tuple[str, int, dict]]):
    fig = make_subplots(rows=len(exp_data_list), cols=1, subplot_titles=[data[0] for data in exp_data_list])
    fig.update_layout(title_text=f"[{fname}] Step Sales v.s. Step Demand")
    for i, (exp_name, line_col, sku_status) in enumerate(exp_data_list):
        for idx, metric, legend in zip(
            [1, 2, 3],
            ["stock_status", "stock_in_transit_status", "stock_ordered_to_distribute_status"],
            ["Stock", "In-Transit", "To-Distribute"]
        ):
            data = sku_status[metric][0, :, status_idx]

            fig.add_trace(
                go.Scatter(
                    x=list(range(len_period)),
                    y=data,
                    mode="lines",
                    legendgroup=exp_name,
                    name=legend,
                    line=LINE_TYPE[f"Fea_{idx}"],
                    showlegend=True if i == 0 else False,
                ),
                row=i + 1,
                col=1,
            )

    st.plotly_chart(fig, use_container_width=True)


def plot_consumer_action_status(
    fname: str, status_idx: int, len_period: int, exp_data_list: List[Tuple[str, int, dict]]
):
    fig = make_subplots(rows=len(exp_data_list), cols=1, subplot_titles=[data[0] for data in exp_data_list])
    fig.update_layout(title_text=f"[{fname}] purchased quantity & received quantity")
    for i, (exp_name, line_col, sku_status) in enumerate(exp_data_list):
        for idx, metric, legend in zip(
            [1, 2],
            ["consumer_purchased", "consumer_received"],
            ["Purchased", "Received"]
        ):
            data = sku_status[metric][0, :, status_idx]

            fig.add_trace(
                go.Scatter(
                    x=list(range(len_period)),
                    y=data,
                    mode="lines",
                    legendgroup=exp_name,
                    name=legend,
                    line=LINE_TYPE[f"Fea_{idx}"],
                    showlegend=True if i == 0 else False,
                ),
                row=i + 1,
                col=1,
            )

    st.plotly_chart(fig, use_container_width=True)


def plot_demand_and_sales(fname: str, status_idx: int, len_period: int, exp_data_list: List[Tuple[str, int, dict]]):
    fig = make_subplots(rows=len(exp_data_list), cols=1, subplot_titles=[data[0] for data in exp_data_list])
    fig.update_layout(title_text=f"[{fname}] Step Sales v.s. Step Demand")
    for i, (exp_name, line_col, sku_status) in enumerate(exp_data_list):
        for idx, metric, legend in zip([1, 2], ["demand_status", "sold_status"], ["Demand", "Sales"]):
            data = sku_status[metric][0, :, status_idx]

            fig.add_trace(
                go.Scatter(
                    x=list(range(len_period)),
                    y=data,
                    mode="lines",
                    legendgroup=exp_name,
                    name=legend,
                    line=LINE_TYPE[f"Fea_{idx}"],
                    showlegend=True if i == 0 else False,
                ),
                row=i + 1,
                col=1,
            )

    st.plotly_chart(fig, use_container_width=True)


def plot_route_network(exp_data: Tuple[str, int, dict]):
    exp_name, line_col, sku_status = exp_data
    upstream_set_dict = sku_status["upstream_set_dict"]

    graph = graphviz.Digraph(
        name=f"Route Network of {exp_name}",
        edge_attr={
            'color': LINE_TYPE[line_col].__getitem__('color'),
        },
        node_attr={
            'shape': 'ellipse',
            'color': LINE_TYPE[line_col].__getitem__('color'),
            'fixedsize': 'true',
            'width': '0.8',
            'height': '0.6',
            'fontsize': '14',
        },
        graph_attr={
            'rankdir': 'LR',
        }
    )
    for f_down, up_list in upstream_set_dict.items():
        for f_up in up_list:
            graph.edge(f_up, f_down)

    st.graphviz_chart(graph, use_container_width=True)


def plot_metric_network(exp_data: Tuple[str, int, dict], metric: str):
    """Valid metric in: ["turnover_rate", "available_rate"]."""
    exp_name, line_col, sku_status = exp_data
    upstream_set_dict = sku_status["upstream_set_dict"]
    metric_dict = sku_status[f"{metric}_dict"]

    graph = graphviz.Digraph(
        name=f"{metric} of {exp_name}",
        edge_attr={
            'color': LINE_TYPE[line_col].__getitem__('color'),
        },
        node_attr={
            'shape': 'ellipse',
            'color': LINE_TYPE[line_col].__getitem__('color'),
            'fixedsize': 'true',
            'width': '0.8',
            'height': '0.6',
            'fontsize': '14',
        },
        graph_attr={
            'rankdir': 'LR',
        }
    )
    for f_down, up_list in upstream_set_dict.items():
        for f_up in up_list:
            if f_up[2] == '_':
                f_up = f"{f_up}-{metric_dict[f_up]}"
            if f_down[2] == '_':
                f_down = f"{f_down}-{metric_dict[f_down]}"
            graph.edge(f_up, f_down)

    st.graphviz_chart(graph, use_container_width=True)


def main():
    st.markdown(MARKDOWN_BODY, unsafe_allow_html=True)

    ############################################################################
    ## Experiment List
    ############################################################################
    st.sidebar.title("Experiment List")

    log_dir = os.path.join(os.path.dirname(__file__), "../../logs")
    exp_list = sorted(os.listdir(log_dir))
    exp_list = ["DUMMY"] + [exp for exp in exp_list if exp[:3] == "SCI"]

    exp_num = st.sidebar.slider("Number of experiments", 1, 4, value=1)
    exp_data_list, by_fname_type_sku, facility_by_name, len_period = set_exp_list(exp_num, exp_list, log_dir)

    ############################################################################
    ## General
    ############################################################################
    st.sidebar.title("General Filter")

    default_facility = [fname for fname in ["CA_3", "CA_4"] if fname in by_fname_type_sku]
    selected_facility = st.sidebar.multiselect("Facility", list(by_fname_type_sku.keys()), default_facility)

    product_type_candidates = set()
    for facility in selected_facility:
        for prefix in by_fname_type_sku[facility].keys():
            product_type_candidates.add(prefix)
    selected_product_type = st.sidebar.selectbox("Product Type", sorted(list(product_type_candidates)), 1)

    sku_candidates = set()
    for facility in selected_facility:
        for sku_name in by_fname_type_sku[facility][selected_product_type].keys():
            sku_candidates.add(sku_name)
    selected_sku_name = st.sidebar.selectbox("SKU Name", sorted(list(sku_candidates)), 0)

    facility_idx_list = [facility_by_name[f_name][0] for f_name in selected_facility]
    status_idx_list = [
        by_fname_type_sku[facility][selected_product_type][selected_sku_name][0]
        for facility in selected_facility
    ]

    exp_name_list = [exp_name for (exp_name, _, _) in exp_data_list]
    selected_exp_names = st.sidebar.multiselect("Experiments", exp_name_list, exp_name_list)

    ############################################################################
    ## Balance Comparison
    ############################################################################
    st.header("Facility Balance")

    if st.checkbox("Team Balance (Stores only)", True):
        plot_team_balance(facility_by_name, len_period, exp_data_list)

    if st.checkbox("Breakdown Balance by Facility", False):
        for f_name, f_idx in zip(selected_facility, facility_idx_list):
            plot_facility_balance(f_name, f_idx, len_period, exp_data_list)

    ############################################################################
    ## Topology Comparison
    ############################################################################
    st.header("Distribution Network Comparison")

    if st.checkbox("Show topologies", False):
        for exp_data in exp_data_list:
            if exp_data[0] in selected_exp_names:
                plot_route_network(exp_data)

    ############################################################################
    ## Metrics Comparison
    ############################################################################
    st.header("Metrics Comparison")

    if st.checkbox("Show turnover rate", False):
        for exp_data in exp_data_list:
            if exp_data[0] in selected_exp_names:
                plot_metric_network(exp_data, "turnover_rate")

    if st.checkbox("Show available rate", False):
        for exp_data in exp_data_list:
            if exp_data[0] in selected_exp_names:
                plot_metric_network(exp_data, "available_rate")

    ############################################################################
    ## SKU Render
    ############################################################################
    st.header("SKU Comparison")

    if st.checkbox("Stock-related Data", False):
        for status_idx, fname in zip(status_idx_list, selected_facility):
            plot_stock_related_status(fname, status_idx, len_period, exp_data_list)

    if st.checkbox("Consumer Action", False):
        for status_idx, fname in zip(status_idx_list, selected_facility):
            plot_consumer_action_status(fname, status_idx, len_period, exp_data_list)

    if st.checkbox("Demand", False):
        for status_idx, fname in zip(status_idx_list, selected_facility):
            plot_single_sku_status(
                f"[{fname}] Step Demand", "demand_status", status_idx, len_period, exp_data_list
            )

    if st.checkbox("Sales v.s. Demand", False):
        for status_idx, fname in zip(status_idx_list, selected_facility):
            plot_demand_and_sales(fname, status_idx, len_period, exp_data_list)

    if st.checkbox("Balance", False):
        for status_idx, fname in zip(status_idx_list, selected_facility):
            plot_single_sku_status(
                f"[{fname}] Step Balance", "balance_status", status_idx, len_period, exp_data_list
            )


if __name__ == "__main__":
    main()
