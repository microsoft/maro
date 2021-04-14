# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from typing import List

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import maro.cli.inspector.dashboard_helper as helper

from .params import CITIBIKEItemOption, GlobalFileNames, GlobalScenarios
from .visualization_choice import CitiBikeIntraViewChoice, PanelViewChoice


def start_citi_bike_dashboard(source_path: str, epoch_num: int, prefix: str):
    """Entrance of Citi_Bike dashboard.

    Expected folder structure of Scenario Citi Bike:
    -source_path
        --epoch_0: Data of each epoch.
            --stations.csv: Record stations' attributes in this file.
            --matrices.csv: Record transfer volume information in this file.
            --stations_summary.csv: Record the summary data of current epoch.
        ………………
        --epoch_{epoch_num-1}
        --manifest.yml: Record basic info like scenario name, name of index_name_mapping file.
        --full_stations.json: Record the relationship between ports' index and name.
        --stations_summary.csv: Record cross-epoch summary data.

    Args:
        source_path (str): The root path of the dumped snapshots data for the corresponding experiment.
        epoch_num (int): Total number of epoches,
            i.e. the total number of data folders since there is a folder per epoch.
        prefix (str): Prefix of data folders.
    """
    if epoch_num > 1:
        option = st.sidebar.selectbox(
            label="Data Type",
            options=PanelViewChoice._member_names_)
        if option == PanelViewChoice.Inter_Epoch.name:
            render_inter_view(source_path, epoch_num)
        elif option == PanelViewChoice.Intra_Epoch.name:
            render_intra_view(source_path, epoch_num, prefix)
        else:
            pass
    else:
        render_intra_view(source_path, epoch_num, prefix)


def render_inter_view(source_path: str, epoch_num: int):
    """Render the cross-epoch infomartion chart of Citi Bike data.

    This part would be displayed only if epoch_num > 1.

    Args:
        source_path (str): The root path of the dumped snapshots data for the corresponding experiment.
        epoch_num (int): Total number of epoches,
            i.e. the total number of data folders since there is a folder per epoch.
    """
    helper.render_h1_title("Citi Bike Inter Epoch Data")
    sample_ratio = helper.get_sample_ratio_selection_list(epoch_num)
    # Get epoch sample number.
    down_pooling_range = helper._get_sampled_epoch_range(epoch_num, sample_ratio)
    attribute_option_candidates = (
        CITIBIKEItemOption.quick_info + CITIBIKEItemOption.requirement_info + CITIBIKEItemOption.station_info
    )
    # Generate data.
    data_summary = helper.read_detail_csv(
        os.path.join(source_path, GlobalFileNames.stations_sum)
    ).iloc[down_pooling_range]
    # Get formula and selected data.
    data_formula = helper.get_filtered_formula_and_data(
        GlobalScenarios.CITI_BIKE, data_summary, attribute_option_candidates
    )
    _generate_inter_view_panel(
        data_formula["data"][data_formula["attribute_option"]],
        down_pooling_range
    )


def _generate_inter_view_panel(data: pd.DataFrame, down_pooling_range: List[float]):
    """Generate inter-view i.e. cross-epoch summary data plot.

    Args:
        data (pd.Dataframe): Original data.
        down_pooling_range (List[float]): Sampling data index list.
    """
    data["Epoch Index"] = list(down_pooling_range)
    data_melt = data.melt("Epoch Index", var_name="Attributes", value_name="Count")

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


def render_intra_view(source_path: str, epoch_num: int, prefix: str):
    """Show Citi Bike intra-view plot.

    Args:
        source_path (str): The root path of the dumped snapshots data for the corresponding experiment.
        epoch_num (int): Total number of epoches,
            i.e. the total number of data folders since there is a folder per epoch.
        prefix (str): Prefix of data folders.
    """
    selected_epoch = 0
    if epoch_num > 1:
        selected_epoch = st.sidebar.select_slider(
            label="Choose an Epoch:",
            options=list(range(0, epoch_num))
        )
    helper.render_h1_title("Citi Bike Intra Epoch Data")
    data_stations = pd.read_csv(os.path.join(source_path, f"{prefix}{selected_epoch}", "stations.csv"))
    view_option = st.sidebar.selectbox(
        label="By station/snapshot:",
        options=CitiBikeIntraViewChoice._member_names_
    )
    stations_num = len(data_stations["id"].unique())
    stations_index = np.arange(stations_num).tolist()
    snapshot_num = len(data_stations["frame_index"].unique())
    snapshots_index = np.arange(snapshot_num).tolist()

    index_name_conversion = helper.read_detail_csv(
        os.path.join(
            source_path,
            GlobalFileNames.name_convert
        )
    )
    attribute_option_candidates = (
        CITIBIKEItemOption.quick_info + CITIBIKEItemOption.requirement_info + CITIBIKEItemOption.station_info
    )
    st.sidebar.markdown("***")

    # Filter by station index.
    # Display the change of snapshot within 1 station.
    render_top_k_summary(source_path, prefix, selected_epoch)
    if view_option == CitiBikeIntraViewChoice.by_station.name:
        _generate_intra_view_by_station(
            data_stations, index_name_conversion, attribute_option_candidates, stations_index, snapshot_num
        )

    # Filter by snapshot index.
    # Display all station information within 1 snapshot.
    elif view_option == CitiBikeIntraViewChoice.by_snapshot.name:
        _generate_intra_view_by_snapshot(
            data_stations, index_name_conversion, attribute_option_candidates,
            snapshots_index, snapshot_num, stations_num
        )


def render_top_k_summary(source_path: str, prefix: str, epoch_index: int):
    """ Show top-k summary plot.

    Args:
        source_path (str): The root path of the dumped snapshots data for the corresponding experiment.
        prefix (str): Prefix of data folders.
        epoch_index (int): The index of selected epoch.
    """
    helper.render_h3_title("Cike Bike Top K")
    data = helper.read_detail_csv(
        os.path.join(
            source_path,
            f"{prefix}{epoch_index}",
            GlobalFileNames.stations_sum
        )
    )
    # Convert index to station name.
    index_name_conversion = helper.read_detail_csv(os.path.join(source_path, GlobalFileNames.name_convert))
    data["station name"] = list(
        map(
            lambda x: index_name_conversion.loc[int(x[9:])][0],
            data["name"]
        )
    )
    # Generate top summary.
    top_number = st.select_slider(
        "Select Top K",
        list(range(1, 6))
    )
    top_attributes = ["bikes", "trip_requirement", "fulfillment", "fulfillment_ratio"]
    for item in top_attributes:
        helper.generate_by_snapshot_top_summary("station name", data, int(top_number), item)


def _generate_intra_view_by_snapshot(
    data_stations: pd.DataFrame, index_name_conversion: pd.DataFrame, attribute_option_candidates: List[str],
    snapshots_index: List[int], snapshot_num: int, stations_num: int
):
    """Show Citi Bike intra-view data by snapshot.

    Args:
        data_stations (pd.Dataframe): Filtered Data.
        index_name_conversion (pd.Dataframe): Relationship between index and name.
        attribute_option_candidates (List[str]): All options for users to choose.
        snapshots_index (List[int]): Sampled snapshot index list.
        snapshot_num (int): Number of snapshots.
        stations_num (int): Number of stations.
    """
    # Get selected snapshot index.
    selected_snapshot = st.sidebar.select_slider(
        label="snapshot index",
        options=snapshots_index
    )
    helper.render_h3_title(f"Snapshot-{selected_snapshot}:  Detail Data")
    # Get according data with selected snapshot.
    data_filtered = data_stations[data_stations["frame_index"] == selected_snapshot]
    # Get increasing rate.
    sample_ratio = helper.get_sample_ratio_selection_list(stations_num)
    # Get sample rate (between 0-1).
    selected_station_sample_ratio = st.sidebar.select_slider(
        label="Station Sampling Ratio",
        options=sample_ratio,
        value=1
    )

    # Get formula input and output.
    data_formula = helper.get_filtered_formula_and_data(
        GlobalScenarios.CITI_BIKE, data_filtered, attribute_option_candidates
    )
    # Get sampled data and get station name.
    down_pooling_sample_list = helper.get_sample_index_list(stations_num, selected_station_sample_ratio)

    attribute_option = data_formula["attribute_option"]
    attribute_option.append("name")
    if selected_station_sample_ratio == 0:
        empty_head = attribute_option
        empty_head.append("Station Name")
        data_filtered = pd.DataFrame(columns=empty_head)
    else:
        data_filtered = data_formula["data"][attribute_option]
        data_filtered = data_filtered.iloc[down_pooling_sample_list]
        data_filtered["name"] = data_filtered["name"].apply(lambda x: int(x[9:]))
        data_filtered["Station Name"] = data_filtered["name"].apply(
            lambda x: index_name_conversion.loc[int(x)]
        )
    data_melt = data_filtered.melt(
        ["Station Name", "name"],
        var_name="Attributes",
        value_name="Count"
    )
    intra_snapshot_line_chart = alt.Chart(data_melt).mark_bar().encode(
        x=alt.X("name:N", axis=alt.Axis(title="Name")),
        y="Count:Q",
        color="Attributes:N",
        tooltip=["Attributes", "Count", "Station Name"],
    ).properties(
        width=700,
        height=380
    )
    st.altair_chart(intra_snapshot_line_chart)


def _generate_intra_view_by_station(
    data_stations: pd.DataFrame, index_name_conversion: pd.DataFrame, attribute_option_candidates: List[str],
    stations_index: List[int], snapshot_num: int
):
    """ Show Citi Bike intra-view data by station.

    Args:
        data_stations (pd.Dataframe): Filtered station data.
        index_name_conversion (pd.Dataframe): Relationship between index and name.
        attribute_option_candidates (List[str]): All options for users to choose.
        stations_index (List[int]):  List of station index.
        snapshot_num (int): Number of snapshots.
    """
    selected_station = st.sidebar.select_slider(
        label="station index",
        options=stations_index
    )
    helper.render_h3_title(index_name_conversion.loc[int(selected_station)][0] + " Detail Data")
    # Filter data by station index.
    data_filtered = data_stations[data_stations["name"] == f"stations_{selected_station}"]
    snapshot_sample_ratio_list = helper.get_sample_ratio_selection_list(snapshot_num)
    selected_snapshot_sample_ratio = st.sidebar.select_slider(
        label="Snapshot Sampling Ratio:",
        options=snapshot_sample_ratio_list,
        value=1
    )
    # Get formula input and output.
    data_formula = helper.get_filtered_formula_and_data(
        GlobalScenarios.CITI_BIKE, data_filtered, attribute_option_candidates
    )

    attribute_option = data_formula["attribute_option"]
    attribute_option.append("frame_index")
    data_filtered = data_formula["data"][attribute_option].reset_index(drop=True)
    down_pooling_sample_list = helper.get_sample_index_list(snapshot_num, selected_snapshot_sample_ratio)
    data_filtered = data_filtered.iloc[down_pooling_sample_list]
    data_filtered.rename(
        columns={"frame_index": "snapshot_index"},
        inplace=True
    )
    data_melt = data_filtered.melt(
        "snapshot_index",
        var_name="Attributes",
        value_name="Count"
    )
    intra_station_line_chart = alt.Chart(data_melt).mark_line().encode(
        x=alt.X("snapshot_index", axis=alt.Axis(title="Snapshot Index")),
        y="Count",
        color="Attributes",
        tooltip=["Attributes", "Count", "snapshot_index"]
    ).properties(
        width=700,
        height=380
    )
    st.altair_chart(intra_station_line_chart)

    intra_station_bar_chart = alt.Chart(data_melt).mark_bar().encode(
        x=alt.X("snapshot_index:N", axis=alt.Axis(title="Snapshot Index")),
        y="Count:Q",
        color="Attributes:N",
        tooltip=["Attributes", "Count", "snapshot_index"]
    ).properties(
        width=700,
        height=380
    )
    st.altair_chart(intra_station_bar_chart)
