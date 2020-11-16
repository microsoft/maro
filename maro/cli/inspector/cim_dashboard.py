import math
import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import maro.cli.inspector.common_helper as common_helper
from maro.cli.inspector.common_params import CIMItemOption, ScenarioDetail
from maro.cli.utils.params import GlobalFilePaths as Gfiles
from maro.cli.utils.params import GlobalScenarios


def generate_down_pooling_sample(down_pooling_num, start_epoch, end_epoch):
    """Generate down pooling list based on original data and down pooling rate.
        This function aims to generate epoch samples.
        No requirements for the sampled data.

    Args:
        down_pooling_num(int): Calculated length of sample data list.
        start_epoch(int): Start index of sample data.
        end_epoch(int): End index of sample data.

    Returns:
        list: sample data list.

    """
    down_pooling_len = math.floor(1 / down_pooling_num)
    down_pooling_range = list(range(start_epoch, end_epoch, down_pooling_len))
    if end_epoch not in down_pooling_range:
        down_pooling_range.append(end_epoch)

    return down_pooling_range


def get_sampled_epoch_range(epoch_num, sample_ratio):
    """For inter plot, generate sampled data list based on range & sample ratio

    Args:
        epoch_num(int): Number of snapshot folders.
        sample_ratio(float): Sampling ratio.
        e.g. If sample_ratio = 0.3, and sample data range = [0,10],
        down_pooling_list = [0, 0.3, 0.6, 0.9]
        down_pooling_range = [0, 3, 6, 9]

    Returns:
        list: list of sampled data index
    """
    start_epoch = st.sidebar.number_input("Start Epoch", 0, epoch_num - 1, 0)
    end_epoch = st.sidebar.number_input("End Epoch", 0, epoch_num - 1, epoch_num - 1)
    down_pooling_num = st.sidebar.select_slider(
        "Epoch Sampling Ratio",
        sample_ratio)
    down_pooling_range = generate_down_pooling_sample(down_pooling_num, start_epoch, end_epoch)
    return down_pooling_range


def show_cim_inter_plot(root_path, epoch_num):
    """Show CIM summary plot.

    Args:
        root_path(str): Data folder path.
        epoch_num(int): Number of snapshot folders
    """
    common_helper.render_h1_title("CIM Inter Epoch Data")
    sample_ratio = common_helper.holder_sample_ratio(epoch_num)
    # get epoch sample num
    down_pooling_range = get_sampled_epoch_range(epoch_num, sample_ratio)
    item_options_all = CIMItemOption.quick_info + CIMItemOption.port_info + CIMItemOption.booking_info
    # generate data
    data = common_helper.read_detail_csv(os.path.join(root_path, Gfiles.ports_sum)).iloc[down_pooling_range]
    data["remaining_space"] = list(
        map(lambda x, y, z: x - y - z, data["capacity"], data["full"], data["empty"]))
    # get formula & selected data
    filtered_data = common_helper.get_filtered_formula_and_data(ScenarioDetail.CIM_Intra, data, item_options_all)
    generate_inter_plot(filtered_data["data"], down_pooling_range)


def generate_inter_plot(data, down_pooling_range):
    """Generate summary plot.
        View info within different epochs.

    Args:
        data(list): Original data.
        down_pooling_range(list): Sampling data index list.
    """
    data["Epoch Index"] = list(down_pooling_range)
    data_long_form = data.melt("Epoch Index", var_name="Attributes", value_name="Count")

    cim_inter_line_chart = alt.Chart(data_long_form).mark_line().encode(
        x="Epoch Index",
        y="Count",
        color="Attributes",
        tooltip=["Attributes", "Count", "Epoch Index"]
    ).properties(
        width=700,
        height=380
    )
    st.altair_chart(cim_inter_line_chart)

    cim_inter_bar_chart = alt.Chart(data_long_form).mark_bar().encode(
        x="Epoch Index:N",
        y="Count:Q",
        color="Attributes:N",
        tooltip=["Attributes", "Count", "Epoch Index"]
    ).properties(
        width=700,
        height=380
    )
    st.altair_chart(cim_inter_bar_chart)


def show_cim_intra_vessel_data(snapshot_index):
    """show vessel info of selected snapshot

    Args:
        snapshot_index(int): Index of selected snapshot folder
    """
    data_vessels = common_helper.read_detail_csv(os.path.join(dir, "vessels.csv"))
    vessels_num = len(data_vessels["name"].unique())
    generate_intra_vessel_by_snapshot(data_vessels, snapshot_index, vessels_num)


def show_cim_intra_by_ports(data_ports, ports_index, name_conversion, item_option_all, ch_info, sf_info, snapshot_num):
    """ show intra data by ports.

    Args:
        data_ports(list): Filtered data.
        ports_index(int):Index of port of current data.
        name_conversion(dataframe): Relationship of index and name.
        item_option_all(list): All options for users to choose.
        ch_info(list): Comprehensive Information. = accumlated data.
        sf_info(list):  Specific Information.
        snapshot_num(int): Number of snapshots on a port.

    """
    port_index = st.sidebar.select_slider(
        "Choose a Port:",
        ports_index)
    port_option = f"ports_{port_index}"
    sample_ratio = common_helper.holder_sample_ratio(snapshot_num)
    snapshot_sample_num = st.sidebar.select_slider("Snapshot Sampling Ratio:", sample_ratio)
    # acc data
    common_helper.render_h1_title("CIM Acc Data")
    common_helper.render_h3_title(f"Port Acc Attributes: {port_index} - {name_conversion.loc[int(port_index)][0]}")
    generate_detail_plot_by_ports(ch_info, data_ports, port_option, snapshot_num, snapshot_sample_num)
    # detail data
    common_helper.render_h1_title("CIM Intra Epoch Data")
    filtered_data = common_helper.get_filtered_formula_and_data(ScenarioDetail.CIM_Inter, data_ports, item_option_all, sf_info)

    common_helper.render_h3_title(
        f"Port Detail Attributes: {port_index} - {name_conversion.loc[int(port_index)][0]}")
    generate_detail_plot_by_ports(
        filtered_data["sf_info"], filtered_data["data"], f"ports_{port_index}",
        snapshot_num,
        snapshot_sample_num, filtered_data["item_option"])


def show_cim_intra_by_snapshot(root_path, option_epoch, data_ports,
                               snapshots_index, name_conversion, item_option_all, ch_info, sf_info, ports_num):
    """

    Args:
        root_path(str): Path of folder.
        option_epoch(int): Index of selected epoch.
        data_ports(list): Filtered data.
        snapshots_index(object): Index of selected snapshot.
        name_conversion(dataframe): Relationship between index and name.
        item_option_all(list): All options for users to choose.
        ch_info(list): Comprehensive Information. = accumlated data.
        sf_info(list):  Specific Information.
        ports_num(int): Number of ports in current snapshot.

    """
    snapshot_index = st.sidebar.select_slider(
        "snapshot index",
        snapshots_index)
    # get sample ratio
    sample_ratio = common_helper.holder_sample_ratio(ports_num)
    usr_ratio = st.sidebar.select_slider("Ports Sample Ratio:", sample_ratio)
    # acc data
    common_helper.render_h1_title("Acc Data")
    show_volume_hot_map(root_path, GlobalScenarios.CIM, option_epoch, snapshot_index)

    common_helper.render_h3_title(f"SnapShot-{snapshot_index}: Port Acc Attributes")
    generate_intra_plot_by_snapshot(ch_info, data_ports, snapshot_index, ports_num, name_conversion, usr_ratio)
    generate_cim_top_summary(data_ports, snapshot_index, ports_num, name_conversion)
    # detail data
    common_helper.render_h1_title("Detail Data")
    show_cim_intra_vessel_data(snapshot_index)

    common_helper.render_h3_title(f"SnapShot-{snapshot_index}: Port Detail Attributes")
    filtered_data = common_helper.get_filtered_formula_and_data(ScenarioDetail.CIM_Inter, data_ports, item_option_all, sf_info)
    generate_intra_plot_by_snapshot(
        filtered_data["sf_info"], filtered_data["data"], snapshot_index,
        ports_num, name_conversion, usr_ratio, filtered_data["item_option"])


def show_cim_intra_plot(root_path, epoch_num):
    """Show CIM detail plot.

    Args:
        root_path (str): Data folder path.
        epoch_num(int) : Number of snapshots
    """
    option_epoch = st.sidebar.select_slider(
        "Choose an Epoch:",
        list(range(0, epoch_num)))
    dir = os.path.join(root_path, f"snapshot_{option_epoch}")
    # get data of selected epoch
    data_ports = common_helper.read_detail_csv(os.path.join(dir, "ports.csv"))
    data_ports["remaining_space"] = list(
        map(lambda x, y, z: x - y - z, data_ports["capacity"], data_ports["full"], data_ports["empty"]))
    # basic data
    ports_num = len(data_ports["name"].unique())
    ports_index = np.arange(ports_num).tolist()
    snapshot_num = len(data_ports["frame_index"].unique())
    snapshots_index = np.arange(snapshot_num).tolist()

    # comprehensive info
    ch_info = CIMItemOption.basic_info + CIMItemOption.acc_info
    common_info = CIMItemOption.booking_info + CIMItemOption.port_info
    # specific info
    sf_info = CIMItemOption.basic_info + common_info
    # item for user to select
    item_option_all = CIMItemOption.quick_info + common_info
    # name conversion
    name_conversion = common_helper.read_detail_csv(os.path.join(root_path, Gfiles.name_convert))

    st.sidebar.markdown("***")
    view_epoch_data_option = ["by ports", "by snapshot"]
    option_2 = st.sidebar.selectbox(
        "By ports/snapshot:",
        view_epoch_data_option)

    if option_2 == "by ports":
        show_cim_intra_by_ports(
            data_ports, ports_index, name_conversion,
            item_option_all, ch_info, sf_info, snapshot_num)

    if option_2 == "by snapshot":
        show_cim_intra_by_snapshot(
            root_path, option_epoch,data_ports, snapshots_index,
            name_conversion, item_option_all, ch_info, sf_info, ports_num)


def generate_cim_top_summary(data, snapshot_index, name_conversion):
    """Generate CIM top 5 summary.

    Args:
        data(list): Data of current snapshot.
        snapshot_index(int): Selected snapshot index.
        name_conversion(dataframe): Relationship between index and name.
    """
    data_acc = data[data["frame_index"] == snapshot_index].reset_index(drop=True)
    data_acc["fulfillment_ratio"] = list(
        map(lambda x, y: float("{:.4f}".format(x / (y + 1 / 1000))), data_acc["acc_fulfillment"],
            data_acc["acc_booking"]))

    data_acc["port name"] = list(map(lambda x: name_conversion.loc[int(x[6:])][0], data_acc["name"]))

    top_attributes = CIMItemOption.acc_info + ["fulfillment_ratio"]
    for item in top_attributes:
        common_helper.generate_by_snapshot_top_summary("port name", data_acc, item, True, snapshot_index)


def generate_intra_vessel_by_snapshot(data_vessels, snapshot_index, vessels_num):
    """Generate vessel detail plot.

    Args:
        data_vessels(list): Data of vessel information within selected snapshot index.
        snapshot_index(int): User-select snapshot index.
        vessels_num(int): Number of vessels.
    """
    common_helper.render_h3_title(f"SnapShot-{snapshot_index}: Vessel Attributes")
    # Get sampled(and down pooling) index
    sample_ratio = common_helper.holder_sample_ratio(vessels_num)
    sample_ratio_res = st.sidebar.select_slider("Vessels Sample Ratio:", sample_ratio)
    down_pooling = list(range(0, vessels_num, math.floor(1 / sample_ratio_res)))

    vessels = data_vessels[data_vessels["frame_index"] == snapshot_index].reset_index(drop=True)
    vessels = vessels[CIMItemOption.vessel_info]
    ss_tmp = pd.DataFrame()
    for index in down_pooling:
        ss_tmp = pd.concat([ss_tmp, vessels[vessels["name"] == f"vessels_{index}"]], axis=0)
    snapshot_filtered = ss_tmp.reset_index(drop=True)

    snapshot_filtered["name"] = snapshot_filtered["name"].apply(lambda x: int(x[8:]))
    snapshot_filtered_long_form = snapshot_filtered.melt("name", var_name="Attributes", value_name="Count")
    vessel_chart_snapshot = alt.Chart(snapshot_filtered_long_form).mark_bar().encode(
        x=alt.X("name:N", axis=alt.Axis(title="Vessel Index")),
        y="Count:Q",
        color="Attributes:N",
        tooltip=["Attributes", "Count", "name"]
    ).properties(
        width=700,
        height=380
    )
    st.altair_chart(vessel_chart_snapshot)


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
        "Dest_Port": np.array(x_axis).ravel(),
        "Start_Port": np.array(y_axis).ravel(),
        "Count": np.array(b).ravel()})
    hot_map = alt.Chart(source).mark_rect().encode(
        x="Dest_Port:O",
        y="Start_Port:O",
        color="Count:Q",
        tooltip=["Dest_Port", "Start_Port", "Count"]
    ).properties(
        width=700,
        height=380
    )
    st.altair_chart(hot_map)


def show_volume_hot_map(root_path, scenario, epoch_index, snapshot_index):
    """Get matrix data and provide entrance to hot map of different scenario.

    Args:
        root_path (str): Data folder path.
        scenario (str): Name of current scenario: CIM.
        epoch_index(int):  Selected epoch index.
        snapshot_index(int): Selected snapshot index.
    """
    matrix_data = pd.read_csv(os.path.join(root_path, f"snapshot_{epoch_index}", "matrices.csv")).loc[snapshot_index]
    if scenario == GlobalScenarios.CIM:
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
    bar_data_long_form = bar_data.melt("snapshot_index", var_name="Attributes", value_name="Count")
    port_line_chart = alt.Chart(bar_data_long_form).mark_line().encode(
        x=alt.X("snapshot_index", axis=alt.Axis(title="Snapshot Index")),
        y="Count",
        color="Attributes",
        tooltip=["Attributes", "Count", "snapshot_index"]
    ).properties(
        width=700,
        height=380
    )
    st.altair_chart(port_line_chart)

    port_bar_chart = alt.Chart(bar_data_long_form).mark_bar().encode(
        x=alt.X("snapshot_index:N", axis=alt.Axis(title="Snapshot Index")),
        y="Count:Q",
        color="Attributes:N",
        tooltip=["Attributes", "Count", "snapshot_index"]
    ).properties(
        width=700,
        height=380)
    st.altair_chart(port_bar_chart)


def generate_intra_plot_by_snapshot(
        info, data, snapshot_index,
        ports_num, name_conversion, sample_ratio, item_option=None):

    """Generate detail plot.
        View info within different snapshot in the same epoch.

    Args:
        info(list): Identifies data at different levels.
                            In this scenario, it is divided into two levels: comprehensive and detail.
                            The list stores the column names that will be extracted at different levels.
        data(list): Filtered data within selected conditions.
        snapshot_index(int): User-select snapshot index.
        ports_num(int): Number of ports.
        name_conversion(dataframe): Relationship between index and name.
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
    snapshot_filtered = snapshot_temp.reset_index(drop=True)

    snapshot_temp["name"] = snapshot_temp["name"].apply(lambda x: int(x[6:]))
    if item_option is not None:
        item_option.append("name")
        snapshot_temp = snapshot_temp[item_option]

    snapshot_filtered["Port Name"] = snapshot_temp["name"].apply(lambda x: name_conversion.loc[int(x)])
    snapshot_filtered_lf = snapshot_filtered.melt(["name", "Port Name"], var_name="Attributes", value_name="Count")
    custom_chart_snapshot = alt.Chart(snapshot_filtered_lf).mark_bar().encode(
        x=alt.X("name:N", axis=alt.Axis(title="Name")),
        y="Count:Q",
        color="Attributes:N",
        tooltip=["Attributes", "Count", "Port Name"]
    ).properties(
        width=700,
        height=380
    )
    st.altair_chart(custom_chart_snapshot)


def start_cim_dashboard(root_path, epoch_num):
    """Entrance of cim dashboard.

    Args:
        root_path (str): Data folder path.
        epoch_num(int) : Number of data folders.
    """
    option = st.sidebar.selectbox(
        "Data Type",
        ("Inter Epoch", "Intra Epoch"))
    if option == "Inter Epoch":
        show_cim_inter_plot(root_path, epoch_num)
    else:
        show_cim_intra_plot(root_path, epoch_num)
