import math
import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import maro.cli.inspector.dashboard_helper as helper

from .params import CITIBIKEItemOption
from .params import GlobalFilePaths
from .params import GlobalScenarios
from .visualization_choice import CitiBikeIntraViewChoice, PanelViewChoice


def start_citi_bike_dashboard(source_path: str, prefix: str):
    """Entrance of citi_bike dashboard.

    Args:
        source_path (str): Data folder path.
        prefix (str): Prefix of data folders.
    """
    option = st.sidebar.selectbox(
        "Data Type",
        PanelViewChoice._member_names_)
    if option == PanelViewChoice.Inter_Epoch.name:
        render_inter_view(source_path)
    elif option == PanelViewChoice.Intra_Epoch.name:
        render_intra_view(source_path, prefix)
    else:
        pass


def render_intra_view(source_path: str, prefix: str):
    """Show citi_bike detail plot.

    Args:
        source_path (str): Data folder path.
        prefix (str): Prefix of data folders.
    """
    helper.render_h1_title("CITI_BIKE Detail Data")
    data_stations = pd.read_csv(os.path.join(source_path, prefix, "stations.csv"))
    view_option = st.sidebar.selectbox(
        "By station/snapshot:",
        CitiBikeIntraViewChoice._member_names_)
    stations_num = len(data_stations["id"].unique())
    stations_index = np.arange(stations_num).tolist()
    snapshot_num = len(data_stations["frame_index"].unique())
    snapshots_index = np.arange(snapshot_num).tolist()

    name_conversion = helper.read_detail_csv(
        os.path.join(
            source_path,
            GlobalFilePaths.name_convert
        )
    )
    option_candidates = CITIBIKEItemOption.quick_info + CITIBIKEItemOption.requirement_info + CITIBIKEItemOption.station_info
    st.sidebar.markdown("***")
    # filter by station index
    # display the change of snapshot within 1 station

    if view_option == CitiBikeIntraViewChoice.by_station.name:
        _generate_inter_view_by_station(
            data_stations, name_conversion, option_candidates, stations_index, snapshot_num
        )
    # filter by snapshot index
    # display all station information within 1 snapshot
    elif view_option == CitiBikeIntraViewChoice.by_snapshot.name:
        _generate_inter_view_by_snapshot(
            data_stations, name_conversion, option_candidates,
            snapshots_index, snapshot_num, stations_num
        )


def render_inter_view(source_path: str):
    """ Show summary plot.

    Args:
        source_path (str): Data folder path.
    """
    helper.render_h1_title("CITI_BIKE Summary Data")
    data = helper.read_detail_csv(
        os.path.join(
            source_path,
            GlobalFilePaths.stations_sum
        )
    )
    # convert index to station name
    name_conversion = helper.read_detail_csv(os.path.join(source_path, GlobalFilePaths.name_convert))
    data["station name"] = list(
        map(
            lambda x: name_conversion.loc[int(x[9:])][0],
            data["name"]
        )
    )
    # generate top summary
    top_number = st.select_slider("Top K", list(range(0, 10)))
    top_attributes = ["bikes", "trip_requirement", "fulfillment", "fulfillment_ratio"]
    for item in top_attributes:
        helper.generate_by_snapshot_top_summary("station name", data, int(top_number), item)


def _generate_inter_view_by_snapshot(
        data_stations: pd.DataFrame, name_conversion: pd.DataFrame, option_candidates: list,
        snapshots_index: list, snapshot_num: int, stations_num: int):
    """Show CITI BIKE detail data by snapshot.

    Args:
        data_stations (dataframe): Filtered Data.
        name_conversion (dataframe): Relationship between index and name.
        option_candidates (list): All options for users to choose.
        snapshots_index (list): Sampled snapshot index.
        snapshot_num (int): Number of snapshots.
        stations_num (int): Number of stations.
    """
    # get selected snapshot index
    snapshot_index = st.sidebar.select_slider(
        "snapshot index",
        snapshots_index)
    helper.render_h3_title(f"Snapshot-{snapshot_index}:  Detail Data")
    # get according data with selected snapshot
    data_filtered = data_stations[data_stations["frame_index"] == snapshot_index]
    # get increasing rate
    sample_ratio = helper.get_holder_sample_ratio(snapshot_num)
    # get sample rate (between 0-1)
    station_sample_num = st.sidebar.select_slider("Snapshot Sampling Ratio", sample_ratio)

    # get formula input & output
    filtered_data = helper.get_filtered_formula_and_data(
        GlobalScenarios.CITI_BIKE, data_filtered, option_candidates)
    # get sampled data & get station name
    down_pooling = list(range(0, stations_num, math.floor(1 / station_sample_num)))

    item_option = filtered_data["item_option"].append("name")
    snapshot_filtered = filtered_data["data"][item_option]
    snapshot_filtered = snapshot_filtered.iloc[down_pooling]
    snapshot_filtered["name"] = snapshot_filtered["name"].apply(lambda x: int(x[9:]))
    snapshot_filtered["Station Name"] = snapshot_filtered["name"].apply(
        lambda x: name_conversion.loc[int(x)]
    )
    data_display = snapshot_filtered.melt(
        ["Station Name", "name"],
        var_name="Attributes",
        value_name="Count"
    )
    snapshot_line_chart = alt.Chart(data_display).mark_bar().encode(
        x=alt.X("name:N", axis=alt.Axis(title="Name")),
        y="Count:Q",
        color="Attributes:N",
        tooltip=["Attributes", "Count", "Station Name"],
    ).properties(
        width=700,
        height=380
    )
    st.altair_chart(snapshot_line_chart)


def _generate_inter_view_by_station(
        data_stations: pd.DataFrame, name_conversion: pd.DataFrame, option_candidates: list,
        stations_index: list, snapshot_num: int):
    """ Show CITI BIKE detail data by station

    Args:
        data_stations (dataframe): Filtered data
        name_conversion (dataframe): Relationship between index and name.
        option_candidates (list): All options for users to choose.
        stations_index (list):  List of station index.
        snapshot_num (int): Number of snapshots.

    """
    station_index = st.sidebar.select_slider(
        "station index",
        stations_index)
    helper.render_h3_title(name_conversion.loc[int(station_index)][0] + " Detail Data")
    # filter data by station index
    data_filtered = data_stations[data_stations["name"] == f"stations_{station_index}"]
    station_sample_ratio = helper.get_holder_sample_ratio(snapshot_num)
    snapshot_sample_num = st.sidebar.select_slider("Snapshot Sampling Ratio:", station_sample_ratio)
    # get formula input & output
    filtered_data = helper.get_filtered_formula_and_data(
        GlobalScenarios.CITI_BIKE, data_filtered, option_candidates)

    item_option = filtered_data["item_option"].append("frame_index")
    station_filtered = filtered_data["data"][item_option].reset_index(drop=True)
    down_pooling = helper.get_snapshot_sample_num(snapshot_num, snapshot_sample_num)
    station_filtered = station_filtered.iloc[down_pooling]
    station_filtered.rename(columns={"frame_index": "snapshot_index"}, inplace=True)
    data_display = station_filtered.melt(
        "snapshot_index",
        var_name="Attributes",
        value_name="Count"
    )
    station_line_chart = alt.Chart(data_display).mark_line().encode(
        x=alt.X("snapshot_index", axis=alt.Axis(title="Snapshot Index")),
        y="Count",
        color="Attributes",
        tooltip=["Attributes", "Count", "snapshot_index"]
    ).properties(
        width=700,
        height=380
    )
    st.altair_chart(station_line_chart)
