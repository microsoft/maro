import math
import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import maro.cli.inspector.common_helper as common_helper
from maro.cli.inspector.common_params import CITIBIKEOption, ScenarioDetail
from maro.cli.utils.params import GlobalFilePaths as Gfiles


def show_citi_bike_detail_by_snapshot(data_stations, name_conversion, item_option_all, snapshots_index, snapshot_num, stations_num):
    """Show CITI BIKE detail data by snapshot.

    Args:
        data_stations(list): Filtered Data.
        name_conversion(dataframe): Relationship between index and name.
        item_option_all(list): All options for users to choose.
        snapshots_index(list): Sampled snapshot index.
        snapshot_num(int): Number of snapshots.
        stations_num(int): Number of stations.
    """
    # get selected snapshot index
    snapshot_index = st.sidebar.select_slider(
        "snapshot index",
        snapshots_index)
    common_helper.render_h3_title(f"Snapshot-{snapshot_index}:  Detail Data")
    # get according data with selected snapshot
    data_filtered = data_stations[data_stations["frame_index"] == snapshot_index]
    # get increasing rate
    sample_ratio = common_helper.holder_sample_ratio(snapshot_num)
    # get sample rate (between 0-1)
    station_sample_num = st.sidebar.select_slider("Snapshot Sampling Ratio", sample_ratio)

    # get formula input & output
    filtered_data = common_helper.get_filtered_formula_and_data(3, data_filtered, item_option_all)
    # get sampled data & get station name
    down_pooling = list(range(0, stations_num, math.floor(1 / station_sample_num)))

    item_option = filtered_data["item_option"].append("name")
    snapshot_filtered = filtered_data["data"][item_option]
    snapshot_filtered = snapshot_filtered.iloc[down_pooling]
    snapshot_filtered["name"] = snapshot_filtered["name"].apply(lambda x: int(x[9:]))
    snapshot_filtered["Station Name"] = snapshot_filtered["name"].apply(lambda x: name_conversion.loc[int(x)])
    snapshot_filtered_long = \
        snapshot_filtered.melt(["Station Name", "name"], var_name="Attributes", value_name="Count")
    snapshot_line_chart = alt.Chart(snapshot_filtered_long).mark_bar().encode(
        x=alt.X("name:N", axis=alt.Axis(title="Name")),
        y="Count:Q",
        color="Attributes:N",
        tooltip=["Attributes", "Count", "Station Name"],
    ).properties(
        width=700,
        height=380
    )
    st.altair_chart(snapshot_line_chart)


def show_citi_bike_detail_by_station(data_stations, name_conversion, item_option_all, stations_index, snapshot_num):
    """ Show CITI BIKE detail data by station

    Args:
        data_stations(list): Filtered data
        name_conversion(dataframe): Relationship between index and name.
        item_option_all(list): All options for users to choose.
        stations_index(list):  List of station index.
        snapshot_num(int): Number of snapshots.

    """
    station_index = st.sidebar.select_slider(
        "station index",
        stations_index)
    common_helper.render_h3_title(name_conversion.loc[int(station_index)][0] + " Detail Data")
    # filter data by station index
    data_filtered = data_stations[data_stations["name"] == f"stations_{station_index}"]
    station_sample_ratio = common_helper.holder_sample_ratio(snapshot_num)
    snapshot_sample_num = st.sidebar.select_slider("Snapshot Sampling Ratio:", station_sample_ratio)
    # get formula input & output
    filtered_data = common_helper.get_filtered_formula_and_data(ScenarioDetail.CITI_BIKE_Detail, data_filtered, item_option_all)

    item_option = filtered_data["item_option"].append("frame_index")
    station_filtered = filtered_data["data"][item_option].reset_index(drop=True)
    down_pooling = common_helper.get_snapshot_sample(snapshot_num, snapshot_sample_num)
    station_filtered = station_filtered.iloc[down_pooling]
    station_filtered.rename(columns={"frame_index": "snapshot_index"}, inplace=True)
    station_filtered_lf = \
        station_filtered.melt("snapshot_index", var_name="Attributes", value_name="Count")
    station_line_chart = alt.Chart(station_filtered_lf).mark_line().encode(
        x=alt.X("snapshot_index", axis=alt.Axis(title="Snapshot Index")),
        y="Count",
        color="Attributes",
        tooltip=["Attributes", "Count", "snapshot_index"]
    ).properties(
        width=700,
        height=380
    )
    st.altair_chart(station_line_chart)


def show_citi_bike_detail_plot(root_path):
    """Show citi_bike detail plot.

    Args:
        root_path (str): Data folder path.
    """

    common_helper.render_h1_title("CITI_BIKE Detail Data")
    data_stations = pd.read_csv(os.path.join(root_path, "snapshot_0", "stations.csv"))
    option = st.sidebar.selectbox(
        "By stations/snapshot:",
        ("by station", "by snapshot"))
    stations_num = len(data_stations["id"].unique())
    stations_index = np.arange(stations_num).tolist()
    snapshot_num = len(data_stations["frame_index"].unique())
    snapshots_index = np.arange(snapshot_num).tolist()
    name_conversion = common_helper.read_detail_csv(os.path.join(root_path, Gfiles.name_convert))

    item_option_all = CITIBIKEOption.quick_info + CITIBIKEOption.requirement_info + CITIBIKEOption.station_info
    st.sidebar.markdown("***")
    # filter by station index
    # display the change of snapshot within 1 station

    if option == "by station":
        show_citi_bike_detail_by_station(data_stations, name_conversion, item_option_all, stations_index, snapshot_num)

    # filter by snapshot index
    # display all station information within 1 snapshot
    if option == "by snapshot":
        show_citi_bike_detail_by_snapshot(data_stations, name_conversion, item_option_all, snapshots_index,
                                          snapshot_num, stations_num)


def show_citi_bike_summary_plot(root_path):
    """ Show summary plot.

    Args:
        root_path (str): Data folder path.

    """
    common_helper.render_H1_title("CITI_BIKE Summary Data")
    data = common_helper.read_detail_csv(os.path.join(root_path, Gfiles.stations_sum))
    # convert index to station name
    name_conversion = common_helper.read_detail_csv(os.path.join(root_path, Gfiles.name_convert))
    data["station name"] = list(map(lambda x: name_conversion.loc[int(x[9:])][0], data["name"]))
    # generate top summary
    top_attributes = ["bikes", "trip_requirement", "fulfillment", "fulfillment_ratio"]
    for item in top_attributes:
        common_helper.generate_by_snapshot_top_summary("station name", data, item, False)


def start_citi_bike_dashboard(root_path):
    """Entrance of citi_bike dashboard.

    Args:
        root_path (str): Data folder path.
    """
    option = st.sidebar.selectbox(
        "Data Type",
        ("Summary", "Detail"))
    if option == "Summary":
        show_citi_bike_summary_plot(root_path)
    else:
        show_citi_bike_detail_plot(root_path)
