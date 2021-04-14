# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import os
from typing import List

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import maro.cli.inspector.dashboard_helper as helper

from .params import CIMItemOption, GlobalFileNames, GlobalScenarios
from .visualization_choice import CIMIntraViewChoice, PanelViewChoice


def start_cim_dashboard(source_path: str, epoch_num: int, prefix: str):
    """Entrance of cim dashboard.

    Expected folder structure of Scenario CIM:
    -source_path
        --epoch_0: Data of each epoch.
            --ports.csv: Record ports' attributes in this file.
            --vessel.csv: Record vessels' attributes in this file.
            --matrices.csv: Record transfer volume information in this file.
        ………………
        --epoch_{epoch_num-1}
        --manifest.yml: Record basic info like scenario name, name of index_name_mapping file.
        --config.yml: Record the relationship between ports' index and name.
        --ports_summary.csv: Record cross-epoch summary data.

    Args:
        source_path (str): The root path of the dumped snapshots data for the corresponding experiment.
        epoch_num (int) : Total number of epoches,
            i.e. the total number of data folders since there is a folder per epoch.
        prefix (str): Prefix of data folders.
    """
    option = st.sidebar.selectbox(
        label="Data Type",
        options=PanelViewChoice._member_names_
    )
    if option == PanelViewChoice.Inter_Epoch.name:
        render_inter_view(source_path, epoch_num)
    elif option == PanelViewChoice.Intra_Epoch.name:
        render_intra_view(source_path, epoch_num, prefix)


def render_inter_view(source_path: str, epoch_num: int):
    """Show CIM inter-view plot.

    Args:
        source_path (str): The root path of the dumped snapshots data for the corresponding experiment.
        epoch_num (int): Total number of epoches,
            i.e. the total number of data folders since there is a folder per epoch.
    """
    helper.render_h1_title("CIM Inter Epoch Data")
    sample_ratio = helper.get_sample_ratio_selection_list(epoch_num)
    # Get epoch sample list.
    down_pooling_range = helper._get_sampled_epoch_range(epoch_num, sample_ratio)
    attribute_option_candidates = (
        CIMItemOption.quick_info + CIMItemOption.port_info + CIMItemOption.booking_info
    )

    # Generate data.
    data = helper.read_detail_csv(os.path.join(source_path, GlobalFileNames.ports_sum)).iloc[down_pooling_range]
    data["remaining_space"] = list(
        map(
            lambda x, y, z: x - y - z,
            data["capacity"],
            data["full"],
            data["empty"]
        )
    )
    # Get formula and selected data.
    data_formula = helper.get_filtered_formula_and_data(GlobalScenarios.CIM, data, attribute_option_candidates)
    _generate_inter_view_panel(
        data_formula["data"][data_formula["attribute_option"]],
        down_pooling_range
    )


def render_intra_view(source_path: str, epoch_num: int, prefix: str):
    """Show CIM intra-view plot.

    Args:
        source_path (str): The root path of the dumped snapshots data for the corresponding experiment.
        epoch_num (int) : Total number of epoches,
            i.e. the total number of data folders since there is a folder per epoch.
        prefix (str):  Prefix of data folders.
    """
    selected_epoch = st.sidebar.select_slider(
        label="Choose an Epoch:",
        options=list(range(0, epoch_num))
    )
    # Get data of selected epoch.
    data_ports = helper.read_detail_csv(os.path.join(source_path, f"{prefix}{selected_epoch}", "ports.csv"))
    data_ports["remaining_space"] = list(
        map(
            lambda x, y, z: x - y - z,
            data_ports["capacity"],
            data_ports["full"],
            data_ports["empty"]
        )
    )
    # Basic data.
    ports_num = len(data_ports["name"].unique())
    ports_index = np.arange(ports_num).tolist()
    snapshot_num = len(data_ports["frame_index"].unique())
    snapshots_index = np.arange(snapshot_num).tolist()

    # Items for user to select.
    attribute_option_candidates = (
        CIMItemOption.quick_info + CIMItemOption.booking_info + CIMItemOption.port_info
    )

    # Name conversion.
    index_name_conversion = helper.read_detail_csv(os.path.join(source_path, GlobalFileNames.name_convert))

    st.sidebar.markdown("***")
    option_view = st.sidebar.selectbox(
        label="By ports/snapshot:",
        options=CIMIntraViewChoice._member_names_
    )

    if option_view == CIMIntraViewChoice.by_port.name:
        _render_intra_view_by_ports(
            data_ports, ports_index, index_name_conversion,
            attribute_option_candidates, snapshot_num
        )
    elif option_view == CIMIntraViewChoice.by_snapshot.name:
        _render_intra_view_by_snapshot(
            source_path, selected_epoch, data_ports, snapshots_index,
            index_name_conversion, attribute_option_candidates, ports_num, prefix
        )


def _generate_inter_view_panel(data: pd.DataFrame, down_pooling_range: List[float]):
    """Generate inter-view plot.

    Args:
        data (pd.Dataframe): Summary(cross-epoch) data.
        down_pooling_range (List[float]): Sampling data index list.
    """
    data["Epoch Index"] = list(down_pooling_range)
    data_melt = data.melt(
        "Epoch Index",
        var_name="Attributes",
        value_name="Count"
    )

    inter_line_chart = alt.Chart(data_melt).mark_line().encode(
        x="Epoch Index",
        y="Count",
        color="Attributes",
        tooltip=["Attributes", "Count", "Epoch Index"]
    ).properties(
        width=700,
        height=380
    )
    st.altair_chart(inter_line_chart)

    inter_bar_chart = alt.Chart(data_melt).mark_bar().encode(
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
    index_name_conversion: pd.DataFrame, attribute_option_candidates: List[str], snapshot_num: int
):
    """ Show intra-view data by ports.

    Args:
        data_ports (pd.Dataframe): Filtered port data.
        ports_index (int):Index of port of current data.
        index_name_conversion (pd.Dataframe): Relationship of index and name.
        attribute_option_candidates (List[str]): All options for users to choose.
        snapshot_num (int): Number of snapshots on a port.
    """
    selected_port = st.sidebar.select_slider(
        label="Choose a Port:",
        options=ports_index
    )
    sample_ratio = helper.get_sample_ratio_selection_list(snapshot_num)
    selected_snapshot_sample_ratio = st.sidebar.select_slider(
        label="Snapshot Sampling Ratio:",
        options=sample_ratio,
        value=1
    )
    # Accumulated data.
    helper.render_h1_title("CIM Accumulated Data")
    helper.render_h3_title(
        f"Port Accumulated Attributes: {selected_port} - {index_name_conversion.loc[int(selected_port)][0]}"
    )
    _generate_intra_panel_accumulated_by_ports(
        data_ports, f"ports_{selected_port}", snapshot_num, selected_snapshot_sample_ratio
    )
    # Detailed data.
    helper.render_h1_title("CIM Detail Data")
    data_formula = helper.get_filtered_formula_and_data(
        GlobalScenarios.CIM, data_ports, attribute_option_candidates
    )

    helper.render_h3_title(
        f"Port Detail Attributes: {selected_port} - {index_name_conversion.loc[int(selected_port)][0]}"
    )
    _generate_intra_panel_by_ports(
        data_formula["data"], f"ports_{selected_port}",
        snapshot_num, selected_snapshot_sample_ratio, data_formula["attribute_option"]
    )


def _render_intra_view_by_snapshot(
    source_path: str, option_epoch: int, data_ports: pd.DataFrame, snapshots_index: List[int],
    index_name_conversion: pd.DataFrame, attribute_option_candidates: List[str], ports_num: int, prefix: str
):
    """ Show intra-view data by snapshot.

    Args:
        source_path (str): The root path of the dumped snapshots data for the corresponding experiment.
        option_epoch (int): Index of selected epoch.
        data_ports (pd.Dataframe): Filtered port data.
        snapshots_index (List[int]): Index of selected snapshot.
        index_name_conversion (pd.Dataframe): Relationship between index and name.
        attribute_option_candidates (List[str]): All options for users to choose.
        ports_num (int): Number of ports in current snapshot.
        prefix (str): Prefix of data folders.
    """
    selected_snapshot = st.sidebar.select_slider(
        label="snapshot index",
        options=snapshots_index
    )
    # Get sample ratio.
    sample_ratio = helper.get_sample_ratio_selection_list(ports_num)
    selected_port_sample_ratio = st.sidebar.select_slider(
        label="Ports Sampling Ratio:",
        options=sample_ratio,
        value=1
    )
    # Accumulated data.
    helper.render_h1_title("Accumulated Data")
    _render_intra_heat_map(source_path, GlobalScenarios.CIM, option_epoch, selected_snapshot, prefix)

    helper.render_h3_title(f"SnapShot-{selected_snapshot}: Port Accumulated Attributes")
    _generate_intra_panel_accumulated_by_snapshot(
        data_ports, selected_snapshot,
        ports_num, index_name_conversion, selected_port_sample_ratio
    )
    _generate_top_k_summary(data_ports, selected_snapshot, index_name_conversion)
    # Detailed data.
    helper.render_h1_title("Detail Data")
    _render_intra_panel_vessel(source_path, prefix, option_epoch, selected_snapshot)

    helper.render_h3_title(f"Snapshot-{selected_snapshot}: Port Detail Attributes")
    data_formula = helper.get_filtered_formula_and_data(
        GlobalScenarios.CIM, data_ports, attribute_option_candidates
    )
    _generate_intra_panel_by_snapshot(
        data_formula["data"], selected_snapshot,
        ports_num, index_name_conversion, selected_port_sample_ratio, data_formula["attribute_option"])


def _generate_intra_panel_by_ports(
    data: pd.DataFrame, option_port_name: str,
    snapshot_num: int, snapshot_sample_num: float, attribute_option: List[str] = None
):
    """Generate intra-view plot.

    View info within different resource holders(In this senario, ports) in the same epoch.
    Change snapshot sampling num freely.

    Args:
        data (pd.Dataframe): Filtered data within selected conditions.
        option_port_name (str): Condition for filtering the name attribute in the data.
        snapshot_num (int): Number of snapshots.
        snapshot_sample_num (float): Number of sampled snapshots.
        attribute_option (List[str]): Translated user-selecteded option.
    """
    if attribute_option is not None:
        attribute_option.append("frame_index")
    else:
        attribute_option = ["frame_index"]
    attribute_temp_option = attribute_option
    attribute_temp_option.append("name")
    data_acc = data[attribute_temp_option]
    down_pooling_sample_list = helper.get_sample_index_list(snapshot_num, snapshot_sample_num)
    port_filtered = data_acc[data_acc["name"] == option_port_name][attribute_option].reset_index(drop=True)
    attribute_option.remove("name")
    data_filtered = port_filtered.loc[down_pooling_sample_list]
    data_filtered = data_filtered[attribute_option]
    data_filtered.rename(
        columns={"frame_index": "snapshot_index"},
        inplace=True
    )
    data_melt = data_filtered.melt(
        "snapshot_index",
        var_name="Attributes",
        value_name="Count"
    )
    port_line_chart = alt.Chart(data_melt).mark_line().encode(
        x=alt.X("snapshot_index", axis=alt.Axis(title="Snapshot Index")),
        y="Count",
        color="Attributes",
        tooltip=["Attributes", "Count", "snapshot_index"]
    ).properties(
        width=700,
        height=380
    )
    st.altair_chart(port_line_chart)

    port_bar_chart = alt.Chart(data_melt).mark_bar().encode(
        x=alt.X("snapshot_index:N", axis=alt.Axis(title="Snapshot Index")),
        y="Count:Q",
        color="Attributes:N",
        tooltip=["Attributes", "Count", "snapshot_index"]
    ).properties(
        width=700,
        height=380)
    st.altair_chart(port_bar_chart)


def _generate_intra_panel_accumulated_by_snapshot(
    data: pd.DataFrame, snapshot_index: int, ports_num: int,
    index_name_conversion: pd.DataFrame, sample_ratio: List[float]
):
    """Generate intra-view accumulated plot by snapshot.

    Args:
        data (pd.Dataframe): Filtered data within selected conditions.
        snapshot_index (int): user-selected snapshot index.
        ports_num (int): Number of ports.
        index_name_conversion (pd.Dataframe): Relationship between index and name.
        sample_ratio (List[float]): Sampled port index list.
    """
    info_selector = CIMItemOption.basic_info + CIMItemOption.acc_info
    data_acc = data[info_selector]
    info_selector.pop(1)
    down_pooling_sample_list = helper.get_sample_index_list(ports_num, sample_ratio)
    snapshot_filtered = data_acc[data_acc["frame_index"] == snapshot_index][info_selector].reset_index(drop=True)
    data_rename = pd.DataFrame(columns=info_selector)
    for index in down_pooling_sample_list:
        data_rename = pd.concat(
            [data_rename, snapshot_filtered[snapshot_filtered["name"] == f"ports_{index}"]],
            axis=0
        )
    data_rename = data_rename.reset_index(drop=True)

    data_rename["name"] = data_rename["name"].apply(lambda x: int(x[6:]))
    data_rename["Port Name"] = data_rename["name"].apply(lambda x: index_name_conversion.loc[int(x)][0])
    data_melt = data_rename.melt(
        ["name", "Port Name"],
        var_name="Attributes",
        value_name="Count"
    )
    intra_bar_chart = alt.Chart(data_melt).mark_bar().encode(
        x=alt.X("name:N", axis=alt.Axis(title="Name")),
        y="Count:Q",
        color="Attributes:N",
        tooltip=["Attributes", "Count", "Port Name"]
    ).properties(
        width=700,
        height=380
    )
    st.altair_chart(intra_bar_chart)


def _generate_intra_panel_accumulated_by_ports(
    data: pd.DataFrame, option_port_name: str, snapshot_num: int, snapshot_sample_num: float
):
    """Generate intra-view accumulated plot by ports.

    Args:
        data (pd.Dataframe): Filtered data within selected conditions.
        option_port_name (str): Condition for filtering the name attribute in the data.
        snapshot_num (int): Number of snapshots.
        snapshot_sample_num (float): Number of sampled snapshots.
    """
    info_selector = CIMItemOption.basic_info + CIMItemOption.acc_info
    data_acc = data[info_selector]
    info_selector.pop(0)
    down_pooling_sample_list = helper.get_sample_index_list(snapshot_num, snapshot_sample_num)
    port_filtered = data_acc[data_acc["name"] == option_port_name][info_selector].reset_index(drop=True)
    port_filtered.rename(
        columns={"frame_index": "snapshot_index"},
        inplace=True
    )

    data_filtered = port_filtered.loc[down_pooling_sample_list]
    data_melt = data_filtered.melt(
        "snapshot_index",
        var_name="Attributes",
        value_name="Count"
    )
    port_line_chart = alt.Chart(data_melt).mark_line().encode(
        x=alt.X("snapshot_index", axis=alt.Axis(title="Snapshot Index")),
        y="Count",
        color="Attributes",
        tooltip=["Attributes", "Count", "snapshot_index"]
    ).properties(
        width=700,
        height=380
    )
    st.altair_chart(port_line_chart)

    port_bar_chart = alt.Chart(data_melt).mark_bar().encode(
        x=alt.X("snapshot_index:N", axis=alt.Axis(title="Snapshot Index")),
        y="Count:Q",
        color="Attributes:N",
        tooltip=["Attributes", "Count", "snapshot_index"]
    ).properties(
        width=700,
        height=380)
    st.altair_chart(port_bar_chart)


def _generate_intra_panel_by_snapshot(
    data: pd.DataFrame, snapshot_index: int, ports_num: int,
    index_name_conversion: pd.DataFrame, sample_ratio: List[float], attribute_option: List[str] = None
):
    """Generate intra-view plot.

    View info within different snapshot in the same epoch.

    Args:
        data (pd.Dataframe): Filtered data within selected conditions.
        snapshot_index (int): user-selected snapshot index.
        ports_num (int): Number of ports.
        index_name_conversion (pd.Dataframe): Relationship between index and name.
        sample_ratio (List[float]): Sampled port index list.
        attribute_option (List[str]): Translated user-selected options.
    """
    if attribute_option is not None:
        attribute_option.append("name")
    else:
        attribute_option = ["name"]
    attribute_temp_option = attribute_option
    attribute_temp_option.append("frame_index")
    data_acc = data[attribute_temp_option]
    down_pooling_sample_list = helper.get_sample_index_list(ports_num, sample_ratio)
    snapshot_filtered = data_acc[data_acc["frame_index"] == snapshot_index][attribute_option].reset_index(drop=True)
    data_rename = pd.DataFrame(columns=attribute_option)
    for index in down_pooling_sample_list:
        data_rename = pd.concat(
            [data_rename, snapshot_filtered[snapshot_filtered["name"] == f"ports_{index}"]],
            axis=0
        )
    data_rename = data_rename.reset_index(drop=True)
    attribute_option.remove("frame_index")
    data_rename["name"] = data_rename["name"].apply(lambda x: int(x[6:]))
    data_rename = data_rename[attribute_option]
    data_rename["Port Name"] = data_rename["name"].apply(lambda x: index_name_conversion.loc[int(x)][0])
    data_melt = data_rename.melt(
        ["name", "Port Name"],
        var_name="Attributes",
        value_name="Count"
    )
    intra_bar_chart = alt.Chart(data_melt).mark_bar().encode(
        x=alt.X("name:N", axis=alt.Axis(title="Name")),
        y="Count:Q",
        color="Attributes:N",
        tooltip=["Attributes", "Count", "Port Name"]
    ).properties(
        width=700,
        height=380
    )
    st.altair_chart(intra_bar_chart)


def _render_intra_panel_vessel(source_path: str, prefix: str, option_epoch: int, snapshot_index: int):
    """Show vessel info of selected snapshot.

    Args:
        source_path (str): The root path of the dumped snapshots data for the corresponding experiment.
        prefix (str): Prefix of data folders.
        option_epoch (int): Selected index of epoch.
        snapshot_index (int): Index of selected snapshot folder.
    """
    data_vessel = helper.read_detail_csv(
        os.path.join(
            source_path,
            f"{prefix}{option_epoch}",
            "vessels.csv"
        )
    )
    vessels_num = len(data_vessel["name"].unique())
    _generate_intra_panel_vessel(data_vessel, snapshot_index, vessels_num)


def _generate_intra_panel_vessel(data_vessel: pd.DataFrame, snapshot_index: int, vessels_num: int):
    """Generate vessel data plot.

    Args:
        data_vessel (pd.Dataframe): Data of vessel information within selected snapshot index.
        snapshot_index (int): User-selected snapshot index.
        vessels_num (int): Number of vessels.
    """
    helper.render_h3_title(f"SnapShot-{snapshot_index}: Vessel Attributes")
    # Get sampled(and down pooling) index.
    sample_ratio = helper.get_sample_ratio_selection_list(vessels_num)
    selected_vessel_sample_ratio = st.sidebar.select_slider(
        label="Vessels Sampling Ratio:",
        options=sample_ratio,
        value=1
    )
    down_pooling_sample_list = helper.get_sample_index_list(vessels_num, selected_vessel_sample_ratio)
    data_vessel = data_vessel[
        data_vessel["frame_index"] == snapshot_index
    ][CIMItemOption.vessel_info].reset_index(drop=True)

    data_rename = pd.DataFrame(columns=CIMItemOption.vessel_info)
    for index in down_pooling_sample_list:
        data_rename = pd.concat(
            [data_rename, data_vessel[data_vessel["name"] == f"vessels_{index}"]],
            axis=0
        )
    data_filtered = data_rename.reset_index(drop=True)
    data_filtered["name"] = data_filtered["name"].apply(lambda x: int(x[8:]))
    data_melt = data_filtered.melt(
        "name",
        var_name="Attributes",
        value_name="Count"
    )
    intra_vessel_bar_chart = alt.Chart(data_melt).mark_bar().encode(
        x=alt.X("name:N", axis=alt.Axis(title="Vessel Index")),
        y="Count:Q",
        color="Attributes:N",
        tooltip=["Attributes", "Count", "name"]
    ).properties(
        width=700,
        height=380
    )
    st.altair_chart(intra_vessel_bar_chart)


def _render_intra_heat_map(
    source_path: str, scenario: GlobalScenarios, epoch_index: int, snapshot_index: int, prefix: str
):
    """Get matrix data and provide entrance to heat map of different scenario.

    Args:
        source_path (str): The root path of the dumped snapshots data for the corresponding experiment.
        scenario (GlobalScenarios): Name of current scenario: CIM.
        epoch_index (int):  Selected epoch index.
        snapshot_index (int): Selected snapshot index.
        prefix (str): Prefix of data folders.
    """
    matrix_data = pd.read_csv(
        os.path.join(
            source_path,
            f"{prefix}{epoch_index}",
            "matrices.csv"
        )
    ).loc[snapshot_index]
    if scenario == GlobalScenarios.CIM:
        helper.render_h3_title(f"snapshot_{snapshot_index}: Accumulated Port Transfer Volume")
        _generate_intra_heat_map(matrix_data["full_on_ports"])


def _generate_intra_heat_map(matrix_data: str):
    """Filter matrix data and generate transfer volume heat map.

    Args:
        matrix_data (str): List of transfer volume within selected snapshot index in string format.
    """
    matrix_data = matrix_data.replace("[", "")
    matrix_data = matrix_data.replace("]", "")
    matrix_data = matrix_data.split()

    matrix_len = int(math.sqrt(len(matrix_data)))
    b = np.array(matrix_data).reshape(matrix_len, matrix_len)

    x_axis = [list(range(0, matrix_len))] * matrix_len
    y_axis = [[row[col] for row in x_axis] for col in range(len(x_axis[0]))]
    # Convert this grid to columnar data expected by Altair.
    data_transfer_volume = pd.DataFrame(
        {
            "Dest_Port": np.array(x_axis).ravel(),
            "Start_Port": np.array(y_axis).ravel(),
            "Count": np.array(b).ravel()
        }
    )
    transfer_volume_heat_map = alt.Chart(data_transfer_volume).mark_rect().encode(
        x="Dest_Port:O",
        y="Start_Port:O",
        color="Count:Q",
        tooltip=["Dest_Port", "Start_Port", "Count"]
    ).properties(
        width=700,
        height=380
    )
    st.altair_chart(transfer_volume_heat_map)


def _generate_top_k_summary(data: pd.DataFrame, snapshot_index: int, index_name_conversion: pd.DataFrame):
    """Generate CIM top k summary.

    Args:
        data (pd.Dataframe): Data of current snapshot.
        snapshot_index (int): Selected snapshot index.
        index_name_conversion (pd.Dataframe): Relationship between index and name.
    """
    data_summary = data[data["frame_index"] == snapshot_index].reset_index(drop=True)
    data_summary["fulfillment_ratio"] = list(
        map(
            lambda x, y: round(x / (y + 1 / 1000), 4),
            data_summary["acc_fulfillment"],
            data_summary["acc_booking"]
        )
    )

    data_summary["port name"] = list(
        map(
            lambda x: index_name_conversion.loc[int(x[6:])][0],
            data_summary["name"]
        )
    )
    helper.render_h3_title("Select Top k:")
    selected_top_number = st.select_slider(
        label="",
        options=list(range(1, 6))
    )
    top_attributes = CIMItemOption.acc_info + ["fulfillment_ratio"]
    for item in top_attributes:
        helper.generate_by_snapshot_top_summary(
            "port name", data_summary, int(selected_top_number), item, snapshot_index
        )
