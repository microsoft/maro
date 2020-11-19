import math
import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import maro.cli.inspector.dashboard_helper as helper

from .params import CIMItemOption
from .params import GlobalFilePaths as Gfiles
from .params import GlobalScenarios
from .visualization_choice import CIMIntraViewChoice, PanelViewChoice


def start_cim_dashboard(source_path: str, epoch_num: int, prefix: str):
    """Entrance of cim dashboard.

    Args:
        source_path (str): Data folder path.
        epoch_num (int) : Number of data folders.
        prefix (str): Prefix of data folders.
    """
    option = st.sidebar.selectbox(
        "Data Type",
        PanelViewChoice._member_names_)
    if option == PanelViewChoice.Inter_Epoch.name:
        render_inter_view(source_path, epoch_num)
    elif option == PanelViewChoice.Intra_Epoch.name:
        render_intra_view(source_path, epoch_num, prefix)
    else:
        pass


def render_inter_view(source_path: str, epoch_num: int):
    """Show CIM summary plot.

    Args:
        source_path (str): Data folder path.
        epoch_num (int): Number of snapshot folders.
    """
    helper.render_h1_title("CIM Inter Epoch Data")
    sample_ratio = helper.get_holder_sample_ratio(epoch_num)
    # get epoch sample num
    down_pooling_range = _get_sampled_epoch_range(epoch_num, sample_ratio)
    option_candidates = CIMItemOption.quick_info + CIMItemOption.port_info + CIMItemOption.booking_info
    # generate data
    data = helper.read_detail_csv(os.path.join(source_path, Gfiles.ports_sum)).iloc[down_pooling_range]
    data["remaining_space"] = list(
        map(lambda x, y, z: x - y - z, data["capacity"], data["full"], data["empty"]))
    # get formula & selected data
    filtered_data = helper.get_filtered_formula_and_data(GlobalScenarios.CIM, data, option_candidates)
    data = filtered_data["data"]
    filtered_option = filtered_data["item_option"]
    _generate_inter_view_panel(data[filtered_option], down_pooling_range)


def render_intra_view(source_path: str, epoch_num: int, prefix: str):
    """Show CIM detail plot.

    Args:
        source_path (str): Data folder path.
        epoch_num (int) : Number of snapshots.
        prefix (str):  Prefix of data folders.
    """
    option_epoch = st.sidebar.select_slider(
        "Choose an Epoch:",
        list(range(0, epoch_num)))
    target_path = os.path.join(source_path, f"{prefix}{option_epoch}")
    # get data of selected epoch
    data_ports = helper.read_detail_csv(os.path.join(target_path, "ports.csv"))
    data_ports["remaining_space"] = list(
        map(lambda x, y, z: x - y - z, data_ports["capacity"], data_ports["full"], data_ports["empty"]))
    # basic data
    ports_num = len(data_ports["name"].unique())
    ports_index = np.arange(ports_num).tolist()
    snapshot_num = len(data_ports["frame_index"].unique())
    snapshots_index = np.arange(snapshot_num).tolist()

    # item for user to select
    option_candidates = CIMItemOption.quick_info + CIMItemOption.booking_info + CIMItemOption.port_info
    # name conversion
    name_conversion = helper.read_detail_csv(os.path.join(source_path, Gfiles.name_convert))

    st.sidebar.markdown("***")
    option_view = st.sidebar.selectbox(
        "By ports/snapshot:",
        CIMIntraViewChoice._member_names_)

    if option_view == CIMIntraViewChoice.by_port.name:
        _render_intra_view_by_ports(
            data_ports, ports_index, name_conversion,
            option_candidates, snapshot_num)
    elif option_view == CIMIntraViewChoice.by_snapshot.name:
        _render_intra_view_by_snapshot(
            source_path, option_epoch, data_ports, snapshots_index,
            name_conversion, option_candidates, ports_num, prefix)
    else:
        pass


def _generate_inter_view_panel(data: pd.DataFrame, down_pooling_range: list):
    """Generate summary plot.
        View info within different epochs.

    Args:
        data (dataframe): Original data.
        down_pooling_range (list): Sampling data index list.
    """
    data["Epoch Index"] = list(down_pooling_range)
    data_long_form = data.melt("Epoch Index", var_name="Attributes", value_name="Count")

    inter_line_chart = alt.Chart(data_long_form).mark_line().encode(
        x="Epoch Index",
        y="Count",
        color="Attributes",
        tooltip=["Attributes", "Count", "Epoch Index"]
    ).properties(
        width=700,
        height=380
    )
    st.altair_chart(inter_line_chart)

    inter_bar_chart = alt.Chart(data_long_form).mark_bar().encode(
        x="Epoch Index:N",
        y="Count:Q",
        color="Attributes:N",
        tooltip=["Attributes", "Count", "Epoch Index"]
    ).properties(
        width=700,
        height=380
    )
    st.altair_chart(inter_bar_chart)


def _render_intra_view_by_ports(
        data_ports: pd.DataFrame, ports_index: int,
        name_conversion: pd.DataFrame, option_candidates: list, snapshot_num: int):
    """ show intra data by ports.

    Args:
        data_ports (dataframe): Filtered data.
        ports_index (int):Index of port of current data.
        name_conversion (dataframe): Relationship of index and name.
        option_candidates (list): All options for users to choose.
        snapshot_num (int): Number of snapshots on a port.
    """
    port_index = st.sidebar.select_slider(
        "Choose a Port:",
        ports_index)
    sample_ratio = helper.get_holder_sample_ratio(snapshot_num)
    snapshot_sample_num = st.sidebar.select_slider("Snapshot Sampling Ratio:", sample_ratio)
    # accumulated data
    helper.render_h1_title("CIM Accumulated Data")
    helper.render_h3_title(
        f"Port Accumulated Attributes: {port_index} - {name_conversion.loc[int(port_index)][0]}")
    _generate_intra_panel_by_ports(
        CIMItemOption.basic_info + CIMItemOption.acc_info,
        data_ports, f"ports_{port_index}", snapshot_num, snapshot_sample_num)
    # detail data
    helper.render_h1_title("CIM Intra Epoch Data")
    filtered_data = helper.get_filtered_formula_and_data(
        GlobalScenarios.CIM, data_ports, option_candidates)

    helper.render_h3_title(
        f"Port Detail Attributes: {port_index} - {name_conversion.loc[int(port_index)][0]}")
    _generate_intra_panel_by_ports(
        CIMItemOption.basic_info + CIMItemOption.booking_info + CIMItemOption.port_info,
        filtered_data["data"], f"ports_{port_index}",
        snapshot_num, snapshot_sample_num, filtered_data["item_option"])


def _render_intra_view_by_snapshot(
        source_path: str, option_epoch: int, data_ports: pd.DataFrame, snapshots_index: list,
        name_conversion: pd.DataFrame, option_candidates: list, ports_num: int, prefix: str):
    """ Show intra-view by snapshot.

    Args:
        source_path (str): Path of folder.
        option_epoch (int): Index of selected epoch.
        data_ports (dataframe): Filtered data.
        snapshots_index (list): Index of selected snapshot.
        name_conversion (dataframe): Relationship between index and name.
        option_candidates (list): All options for users to choose.
        ports_num (int): Number of ports in current snapshot.
        prefix (str): Prefix of data folders.
    """
    snapshot_index = st.sidebar.select_slider(
        "snapshot index",
        snapshots_index)
    # get sample ratio
    sample_ratio = helper.get_holder_sample_ratio(ports_num)
    usr_ratio = st.sidebar.select_slider("Ports Sample Ratio:", sample_ratio)
    # acc data
    helper.render_h1_title("Accumulated Data")
    _render_intra_heat_map(source_path, GlobalScenarios.CIM, option_epoch, snapshot_index, prefix)

    helper.render_h3_title(f"SnapShot-{snapshot_index}: Port Accumulated Attributes")
    _generate_intra_panel_by_snapshot(
        CIMItemOption.basic_info + CIMItemOption.acc_info, data_ports, snapshot_index,
        ports_num, name_conversion, usr_ratio)
    _generate_top_k_summary(data_ports, snapshot_index, name_conversion)
    # detail data
    helper.render_h1_title("Detail Data")
    _render_intra_panel_vessel(source_path, prefix, option_epoch, snapshot_index)

    helper.render_h3_title(f"SnapShot-{snapshot_index}: Port Detail Attributes")
    filtered_data = helper.get_filtered_formula_and_data(
        GlobalScenarios.CIM, data_ports, option_candidates)
    _generate_intra_panel_by_snapshot(
        CIMItemOption.basic_info + CIMItemOption.booking_info + CIMItemOption.port_info,
        filtered_data["data"], snapshot_index,
        ports_num, name_conversion, usr_ratio, filtered_data["item_option"])


def _generate_intra_panel_by_ports(
        info_selector: list, data: pd.DataFrame, option_port_name: str,
        snapshot_num: int, snapshot_sample_num: float, item_option: list = None):
    """Generate detail plot.
        View info within different holders(ports,stations,etc) in the same epoch.
        Change snapshot sampling num freely.
    Args:
        info_selector (list): Identifies data at different levels.
                            In this scenario, it is divided into two levels: comprehensive and detail.
                            The list stores the column names that will be extracted at different levels.
        data (dataframe): Filtered data within selected conditions.
        option_port_name (str): Condition for filtering the name attribute in the data.
        snapshot_num (int): Number of snapshots
        snapshot_sample_num (float): Number of sampled snapshots
        item_option (list): Translated user-select option
    """
    data_acc = data[info_selector]
    # delete parameter:name
    info_selector.pop(0)
    down_pooling = helper.get_snapshot_sample_num(snapshot_num, snapshot_sample_num)
    port_filtered = data_acc[data_acc["name"] == option_port_name][info_selector].reset_index(drop=True)
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


def _generate_intra_panel_by_snapshot(
        info: list, data: pd.DataFrame, snapshot_index: int,
        ports_num: int, name_conversion: pd.DataFrame, sample_ratio: list, item_option: list = None):
    """Generate detail plot.
        View info within different snapshot in the same epoch.

    Args:
        info (list): Identifies data at different levels.
                            In this scenario, it is divided into two levels: comprehensive and detail.
                            The list stores the column names that will be extracted at different levels.
        data (dataframe): Filtered data within selected conditions.
        snapshot_index (int): User-select snapshot index.
        ports_num (int): Number of ports.
        name_conversion (dataframe): Relationship between index and name.
        sample_ratio (list): Sampled port index list.
        item_option (list): Translated user-select options.
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


def _render_intra_panel_vessel(source_path: str, prefix: str, option_epoch: int, snapshot_index: int):
    """show vessel info of selected snapshot

    Args:
        source_path (str): Root path of data folders.
        prefix (str): Prefix of data folders.
        option_epoch (int): Selected index of epoch.
        snapshot_index (int): Index of selected snapshot folder.
    """
    data_vessels = helper.read_detail_csv(os.path.join(source_path, f"{prefix}{option_epoch}", "vessels.csv"))
    vessels_num = len(data_vessels["name"].unique())
    _generate_intra_panel_vessel(data_vessels, snapshot_index, vessels_num)


def _generate_intra_panel_vessel(data_vessels: pd.DataFrame, snapshot_index: int, vessels_num: int):
    """Generate vessel detail plot.

    Args:
        data_vessels (dataframe): Data of vessel information within selected snapshot index.
        snapshot_index (int): User-select snapshot index.
        vessels_num (int): Number of vessels.
    """
    helper.render_h3_title(f"SnapShot-{snapshot_index}: Vessel Attributes")
    # Get sampled(and down pooling) index
    sample_ratio = helper.get_holder_sample_ratio(vessels_num)
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


def _render_intra_heat_map(source_path: str, scenario: enumerate, epoch_index: int, snapshot_index: int, prefix: str):
    """Get matrix data and provide entrance to hot map of different scenario.

    Args:
        source_path (str): Data folder path.
        scenario (enumerate): Name of current scenario: CIM.
        epoch_index (int):  Selected epoch index.
        snapshot_index (int): Selected snapshot index.
        prefix (str): Prefix of data folders.
    """
    matrix_data = pd.read_csv(os.path.join(source_path, f"{prefix}{epoch_index}", "matrices.csv")).loc[snapshot_index]
    if scenario == GlobalScenarios.CIM:
        helper.render_h3_title(f"snapshot_{snapshot_index}: Accumulated Port Transfer Volume")
        _generate_intra_heat_map(matrix_data["full_on_ports"])


def _generate_intra_heat_map(matrix_data: str):
    """Filter matrix data and generate transfer volume hot map.

    Args:
        matrix_data (str): list of transfer volume within selected snapshot index in str format.
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


def _generate_top_k_summary(data: pd.DataFrame, snapshot_index: int, name_conversion: pd.DataFrame):
    """Generate CIM top 5 summary.

    Args:
        data (dataframe): Data of current snapshot.
        snapshot_index (int): Selected snapshot index.
        name_conversion (dataframe): Relationship between index and name.
    """
    data_acc = data[data["frame_index"] == snapshot_index].reset_index(drop=True)
    data_acc["fulfillment_ratio"] = list(
        map(lambda x, y: float("{:.4f}".format(x / (y + 1 / 1000))), data_acc["acc_fulfillment"],
            data_acc["acc_booking"]))

    data_acc["port name"] = list(map(lambda x: name_conversion.loc[int(x[6:])][0], data_acc["name"]))
    helper.render_h3_title("Select Top k")
    top_number = st.select_slider("", list(range(0, 10)))
    top_attributes = CIMItemOption.acc_info + ["fulfillment_ratio"]
    for item in top_attributes:
        helper.generate_by_snapshot_top_summary("port name", data_acc, int(top_number), item, snapshot_index)


def _generate_down_pooling_sample(down_pooling_num: int, start_epoch: int, end_epoch: int) -> list:
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


def _get_sampled_epoch_range(epoch_num: int, sample_ratio: float) -> list:
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
    down_pooling_range = _generate_down_pooling_sample(down_pooling_num, start_epoch, end_epoch)
    return down_pooling_range
