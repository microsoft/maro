import math
import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import maro.cli.inspector.common_helper as common_helper
from maro.cli.utils.params import GlobalPaths

NAME_CONVERSION_PATH = GlobalPaths.MARO_INSPECTOR_FILE_PATH["name_conversion_path"]
PORTS_FILE_PATH = GlobalPaths.MARO_INSPECTOR_FILE_PATH["ports_file_path"]
VESSELS_FILE_PATH = GlobalPaths.MARO_INSPECTOR_FILE_PATH["vessels_file_path"]


def generate_down_pooling_sample(down_pooling_len, start_epoch, end_epoch):
    """Generate down pooling list based on origin data and down pooling rate.
        This generate downpooling sample function is to generate epoch samples.
        No requirements for the sampled data.

    Args:
        down_pooling_len(int): Calculated length of sample data list.
        start_epoch(int): Start index of sample data.
        end_epoch(int): End index of sample data.

    Returns:
        list: sample data list.

    """
    down_pooling_range = list(range(start_epoch, end_epoch, down_pooling_len))
    if end_epoch not in down_pooling_range:
        down_pooling_range.append(end_epoch)
    return down_pooling_range


# entrance of summary plot
def show_cim_summary_plot(ROOT_PATH):
    """Show cim summary plot.

    Args:
        ROOT_PATH (str): Data folder path.
    """
    common_helper.render_H1_title("CIM Summary Data")
    dirs = os.listdir(ROOT_PATH)
    epoch_num = common_helper.get_epoch_num(len(dirs), ROOT_PATH)
    sample_ratio = common_helper.holder_sample_ratio(epoch_num)
    start_epoch = st.sidebar.number_input("Start Epoch", 0, epoch_num - 1, 0)
    end_epoch = st.sidebar.number_input("End Epoch", 0, epoch_num - 1, epoch_num - 1)

    down_pooling_num = st.sidebar.select_slider(
        "Epoch Sampling Ratio",
        sample_ratio)
    down_pooling_len = math.floor(1 / down_pooling_num)
    down_pooling_range = generate_down_pooling_sample(down_pooling_len, start_epoch, end_epoch)
    item_options_all = ["All", "Booking Info", "Port Info",
                        "shortage", "booking",
                        "fulfillment", "on_shipper",
                        "on_consignee", "capacity", "full", "empty"]
    data = common_helper.read_detail_csv(os.path.join(ROOT_PATH, PORTS_FILE_PATH))
    data = data.iloc[down_pooling_range]
    data_genera = common_helper.formula_define(data)
    if data_genera is not None:
        data = data_genera["data_after"]
        item_options_all.append(data_genera["name"])
    item_option = st.multiselect(
        " ", item_options_all, item_options_all)
    item_option = get_CIM_item_option(item_option, item_options_all)
    data = data[item_option]
    generate_summary_plot(item_option, data, down_pooling_range)


def generate_summary_plot(item_option, data, down_pooling_range):
    """Generate summary plot.
        View info within different epochs.

    Args:
        item_option(list): User-select attributes to be displayed.
        data(list): Origin data.
        down_pooling_range(list): Sampling data index list.
    """
    data["epoch index"] = list(down_pooling_range)
    data_long_form = data.melt("epoch index", var_name="Attributes", value_name="count")
    custom_chart_port = alt.Chart(data_long_form).mark_line().encode(
        x="epoch index",
        y="count",
        color="Attributes",
        tooltip=["Attributes", "count", "epoch index"]
    ).properties(
        width=700,
        height=380
    )
    st.altair_chart(custom_chart_port)

    custom_chart_port_bar = alt.Chart(data_long_form).mark_bar().encode(
        x="epoch index:N",
        y="count:Q",
        color="Attributes:N",
        tooltip=["Attributes", "count", "epoch index"]
    ).properties(
        width=700,
        height=380
    )
    st.altair_chart(custom_chart_port_bar)


def get_CIM_item_option(item_option, item_option_all):
    """Convert selected CIM option into column.

    Args:
        item_option(list): User selected option list.
        item_option_all(list): Pre-defined option list.

    Returns:
        list: translated users" option.

    """
    item_option_res = []
    for item in item_option:
        if item == "All":
            item_option_all.remove("All")
            item_option_all.remove("Booking Info")
            item_option_all.remove("Port Info")
            item_option_res = item_option_all
            break
        elif item == "Booking Info":
            item_option_res.append("booking")
            item_option_res.append("shortage")
            item_option_res.append("fulfillment")
        elif item == "Port Info":
            item_option_res.append("full")
            item_option_res.append("empty")
            item_option_res.append("on_shipper")
            item_option_res.append("on_consignee")
        else:
            item_option_res.append(item)
    return item_option_res


def show_cim_detail_plot(ROOT_PATH):
    """Show cim detail plot.

    Args:
        ROOT_PATH (str): Data folder path.
    """
    dirs = os.listdir(ROOT_PATH)
    epoch_num = common_helper.get_epoch_num(len(dirs), ROOT_PATH)

    option_epoch = st.sidebar.select_slider(
        "Choose an Epoch:",
        list(range(0, epoch_num)))
    dir = os.path.join(ROOT_PATH, f"snapshot_{option_epoch}")
    # data_ports = feather_read_data(os.path.join(dir, "ports_feather"))
    data_ports = common_helper.read_detail_csv(os.path.join(dir, "ports.csv"))
    data_ports["remaining_space"] = list(
        map(lambda x, y, z: x - y - z, data_ports["capacity"], data_ports["full"], data_ports["empty"]))
    ports_num = len(data_ports["name"].unique())
    ports_index = np.arange(ports_num).tolist()
    snapshot_num = len(data_ports["frame_index"].unique())
    snapshots_index = np.arange(snapshot_num).tolist()

    st.sidebar.markdown("***")
    option_2 = st.sidebar.selectbox(
        "By ports/snapshot:",
        ("by ports", "by snapshot"))
    comprehensive_info = ["name", "frame_index", "acc_shortage", "acc_booking", "acc_fulfillment"]
    specific_info = ["name", "frame_index", "shortage", "booking",
                        "fulfillment", "on_shipper", "on_consignee",
                        "capacity", "full", "empty", "remaining_space"]
    item_option_all = ["All", "Booking Info", "Port Info",
                            "shortage", "booking", "fulfillment", "on_shipper",
                            "on_consignee", "capacity", "full", "empty", "remaining_space"]
    if option_2 == "by ports":
        port_index = st.sidebar.select_slider(
            "Choose a Port:",
            ports_index)
        str_port_option = f"ports_{port_index}"
        name_conversion = common_helper.read_detail_csv(os.path.join(ROOT_PATH, NAME_CONVERSION_PATH))
        sample_ratio = common_helper.holder_sample_ratio(snapshot_num)
        snapshot_sample_num = st.sidebar.select_slider("Snapshot Sampling Ratio:", sample_ratio)

        common_helper.render_H1_title("CIM Acc Data")
        common_helper.render_H3_title(f"Port Acc Attributes: {port_index} - {name_conversion.loc[int(port_index)][0]}")
        generate_detail_plot_by_ports(comprehensive_info, data_ports,
                                        str_port_option, snapshot_num, snapshot_sample_num)
        common_helper.render_H1_title("CIM Detail Data")
        data_genera = common_helper.formula_define(data_ports)
        if data_genera is not None:
            specific_info.append(data_genera["name"])
            data_ports = data_genera["data_after"]
            item_option_all.append(data_genera["name"])
        item_option = st.multiselect(
            " ", item_option_all, item_option_all)
        item_option = get_CIM_item_option(item_option, item_option_all)
        str_temp = f"ports_{port_index}"
        common_helper.render_H3_title(
            f"Port Detail Attributes: {port_index} - {name_conversion.loc[int(port_index)][0]}")
        generate_detail_plot_by_ports(
            specific_info, data_ports, str_temp,
            snapshot_num,
            snapshot_sample_num, item_option)
    if option_2 == "by snapshot":
        CONVER = os.path.join(ROOT_PATH, NAME_CONVERSION_PATH)
        snapshot_index = st.sidebar.select_slider(
            "snapshot index",
            snapshots_index)
        sample_ratio = common_helper.holder_sample_ratio(ports_num)
        sample_ratio_res = st.sidebar.select_slider("Ports Sample Ratio:", sample_ratio)
        common_helper.render_H1_title("Acc Data")
        show_volume_hot_map(ROOT_PATH, "cim", option_epoch, snapshot_index)
        common_helper.render_H3_title(f"SnapShot-{snapshot_index}: Port Acc Attributes")
        generate_detail_plot_by_snapshot(comprehensive_info, data_ports, snapshot_index, ports_num,
                                            CONVER, sample_ratio_res)
        generate_cim_top_summary(data_ports, snapshot_index, ports_num, os.path.join(ROOT_PATH, NAME_CONVERSION_PATH))
        common_helper.render_H1_title("Detail Data")
        data_vessels = common_helper.read_detail_csv(os.path.join(dir, "vessels.csv"))
        vessels_num = len(data_vessels["name"].unique())
        generate_detail_vessel_by_snapshot(data_vessels, snapshot_index, vessels_num)
        common_helper.render_H3_title(f"SnapShot-{snapshot_index}: Port Detail Attributes")
        data_genera = common_helper.formula_define(data_ports)
        if data_genera is not None:
            specific_info.append(data_genera["name"])
            data_ports = data_genera["data_after"]
            item_option_all.append(data_genera["name"])
        item_option = st.multiselect(" ", item_option_all, item_option_all)
        item_option = get_CIM_item_option(item_option, item_option_all)
        generate_detail_plot_by_snapshot(specific_info, data_ports, snapshot_index,
                                            ports_num, CONVER, sample_ratio_res, item_option)


def generate_cim_top_summary(data, snapshot_index, ports_num, CONVER_PATH):
    """Generate CIM top 5 summary.

    Args:
        data(list): Data of current snapshot.
        snapshot_index(int): Selected snapshot index.
        ports_num(int): Number of ports.
    """
    data_acc = data[data["frame_index"] == snapshot_index].reset_index(drop=True)
    data_acc["fulfillment_ratio"] = list(
        map(lambda x, y: float("{:.4f}".format(x / (y + 1 / 1000))), data_acc["acc_fulfillment"],
            data_acc["acc_booking"]))
    name_conversion = common_helper.read_detail_csv(CONVER_PATH)
    data_acc["port name"] = list(map(lambda x: name_conversion.loc[int(x[6:])][0], data_acc["name"]))
    df_booking = data_acc[["port name", "acc_booking"]].sort_values(by="acc_booking", ascending=False).head(5)
    df_fulfillment = data_acc[["port name", "acc_fulfillment"]].sort_values(by="acc_fulfillment",
                                                                            ascending=False).head(5)
    df_shortage = data_acc[["port name", "acc_shortage"]].sort_values(by="acc_shortage", ascending=False).head(5)
    df_ratio = data_acc[["port name", "fulfillment_ratio"]].sort_values(by="fulfillment_ratio", ascending=False).head(5)
    common_helper.generate_by_snapshot_top_summary("port name", df_booking, "acc_booking", True, snapshot_index)
    common_helper.generate_by_snapshot_top_summary("port name", df_fulfillment, "acc_fulfillment", True, snapshot_index)
    common_helper.generate_by_snapshot_top_summary("port name", df_shortage, "acc_shortage", True, snapshot_index)
    common_helper.generate_by_snapshot_top_summary("port name", df_ratio, "fulfillment_ratio", True, snapshot_index)


def generate_detail_vessel_by_snapshot(data_vessels, snapshot_index, vessels_num):
    """Generate vessel detail plot.

    Args:
        data_vessels(list): Data of vessel information within selected snapshot index.
        snapshot_index(int): User-select snapshot index.
        vessels_num(int): Number of vessels.
    """
    common_helper.render_H3_title(f"SnapShot-{snapshot_index}: Vessel Attributes")
    sample_ratio = common_helper.holder_sample_ratio(vessels_num)
    sample_ratio_res = st.sidebar.select_slider("Vessels Sample Ratio:", sample_ratio)
    down_pooling = list(range(0, vessels_num, math.floor(1 / sample_ratio_res)))
    ss_filtered = data_vessels[data_vessels["frame_index"] == snapshot_index].reset_index(drop=True)
    ss_filtered = ss_filtered[["capacity", "full", "empty", "remaining_space", "name"]]
    ss_tmp = pd.DataFrame()
    for index in down_pooling:
        ss_tmp = pd.concat([ss_tmp, ss_filtered[ss_filtered["name"] == f"vessels_{index}"]], axis=0)
    snapshot_filtered = ss_tmp.reset_index(drop=True)
    snapshot_filtered["name"] = snapshot_filtered["name"].apply(lambda x: int(x[8:]))
    snapshot_filtered_long_form = snapshot_filtered.melt("name", var_name="Attributes", value_name="count")
    custom_chart_snapshot = alt.Chart(snapshot_filtered_long_form).mark_bar().encode(
        x=alt.X("name:N", axis=alt.Axis(title="vessel name")),
        y="count:Q",
        color="Attributes:N",
        tooltip=["Attributes", "count", "name"]
    ).properties(
        width=700,
        height=380
    )
    st.altair_chart(custom_chart_snapshot)


def generate_hot_map(matrix_data):
    """Filter matrix data and generate transfer volume hot map.

    Args:
        matrix_data(str): list of transfer volume within selected snapshot index in str format.
    """
    matrix_data = matrix_data.replace("[", "")
    matrix_data = matrix_data.replace("]", "")
    matrix_data = matrix_data.split()
    matrix_len = int(math.sqrt(len(matrix_data)))
    b = np.array(matrix_data).reshape(matrix_len, matrix_len)
    x_axis_single = list(range(0, matrix_len))
    x_axis = [x_axis_single] * matrix_len
    y_axis = [[row[col] for row in x_axis] for col in range(len(x_axis[0]))]
    # Convert this grid to columnar data expected by Altair
    source = pd.DataFrame({
        "dest_port": np.array(x_axis).ravel(),
        "start_port": np.array(y_axis).ravel(),
        "count": np.array(b).ravel()})
    chart = alt.Chart(source).mark_rect().encode(
        x="dest_port:O",
        y="start_port:O",
        color="count:Q",
        tooltip=["dest_port", "start_port", "count"]
    ).properties(
        width=700,
        height=380
    )
    st.altair_chart(chart)


def show_volume_hot_map(ROOT_PATH, scenario, epoch_index, snapshot_index):
    """Get matrix data and provide entrance to hot map of different scenario.

    Args:
        scenario (str): Name of current scenario: CIM.
        ROOT_PATH (str): Data folder path.
        epoch_index(int):  Selected epoch index.
        snapshot_index(int): Selected snapshot index.
    """
    matrix_data = pd.read_csv(os.path.join(ROOT_PATH, f"snapshot_{epoch_index}", "matrices.csv")).loc[snapshot_index]
    if scenario == "cim":
        common_helper.render_H3_title(f"SnapShot-{snapshot_index}: Acc Port Transfer Volume")
        generate_hot_map(matrix_data["full_on_ports"])


def generate_detail_plot_by_ports(info_selector, data, str_temp, snapshot_num, snapshot_sample_num, item_option=None):
    """Generate detail plot.
        View info within different holders(ports,stations,etc) in the same epoch.
        Change snapshot sampling num freely.
    Args:
        info_selector(list): Identifies data at different levels.
                            In this scenario, it is divided into two levels: comprehensive and detail.
                            The list stores the column names that will be extracted at different levels.
        data(list): Filtered data within selected conditions.
        str_temp(str): Condition for filtering the name attribute in the data.
        snapshot_num(int): Number of snapshots
        snapshot_sample_num(int): Number of sampled snapshots
        item_option(list): Translated user-select option
    """
    data_acc = data[info_selector]
    # delete parameter:name
    info_selector.pop(0)
    down_pooling = common_helper.get_snapshot_sample(snapshot_num, snapshot_sample_num)
    port_filtered = data_acc[data_acc["name"] == str_temp][info_selector].reset_index(drop=True)
    port_filtered.rename(columns={"frame_index": "snapshot_index"}, inplace=True)

    bar_data = port_filtered.loc[down_pooling]
    if item_option is not None:
        item_option.append("snapshot_index")
        bar_data = bar_data[item_option]
    bar_data_long_form = bar_data.melt("snapshot_index", var_name="Attributes", value_name="count")
    custom_chart_port = alt.Chart(bar_data_long_form).mark_line().encode(
        x="snapshot_index",
        y="count",
        color="Attributes",
        tooltip=["Attributes", "count", "snapshot_index"]
    ).properties(
        width=700,
        height=380
    )
    st.altair_chart(custom_chart_port)

    custom_bar_chart = alt.Chart(bar_data_long_form).mark_bar().encode(
        x="snapshot_index:N",
        y="count:Q",
        color="Attributes:N",
        tooltip=["Attributes", "count", "snapshot_index"]
    ).properties(
        width=700,
        height=380)
    st.altair_chart(custom_bar_chart)


def generate_detail_plot_by_snapshot(info, data, snapshot_index, ports_num, CONVER, sample_ratio, item_option=None):
    """Generate detail plot.
        View info within different snapshot in the same epoch.

    Args:
        info(list): Identifies data at different levels.
                            In this scenario, it is divided into two levels: comprehensive and detail.
                            The list stores the column names that will be extracted at different levels.
        data(list): Filtered data within selected conditions.
        snapshot_index(int): User-select snapshot index.
        ports_num(int): Number of ports.
        CONVER(str): Path of name conversion file.
        sample_ratio(list): Sampled port index list.
        item_option(list): Translated user-select options.
    """
    data_acc = data[info]
    # delete parameter:frame_index
    info.pop(1)
    down_pooling = list(range(0, ports_num, math.floor(1 / sample_ratio)))
    snapshot_filtered = data_acc[data_acc["frame_index"] == snapshot_index][info].reset_index(drop=True)
    snapshot_temp = pd.DataFrame(columns=info)
    for index in down_pooling:
        snapshot_temp = pd.concat(
            [snapshot_temp, snapshot_filtered[snapshot_filtered["name"] == f"ports_{index}"]], axis=0)
    snapshot_temp["name"] = snapshot_temp["name"].apply(lambda x: int(x[6:]))
    snapshot_filtered = snapshot_temp.reset_index(drop=True)
    if item_option is not None:
        item_option.append("name")
        snapshot_filtered = snapshot_filtered[item_option]

    name_conversion = common_helper.read_detail_csv(CONVER)
    snapshot_filtered["port name"] = snapshot_filtered["name"].apply(lambda x: name_conversion.loc[int(x)])
    snapshot_filtered_lf = snapshot_filtered.melt(["name", "port name"], var_name="Attributes", value_name="count")
    custom_chart_snapshot = alt.Chart(snapshot_filtered_lf).mark_bar().encode(
        x="name:N",
        y="count:Q",
        color="Attributes:N",
        tooltip=["Attributes", "count", "port name"]
    ).properties(
        width=700,
        height=380
    )
    st.altair_chart(custom_chart_snapshot)


def start_cim_dashboard(ROOT_PATH):
    """Entrance of cim dashboard.

    Args:
        ROOT_PATH (str): Data folder path.
    """
    option = st.sidebar.selectbox(
        "Data Type",
        ("Extro Epoch", "Intra Epoch"))
    if option == "Extro Epoch":
        show_cim_summary_plot(ROOT_PATH)
    else:
        show_cim_detail_plot(ROOT_PATH)
