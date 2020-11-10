import streamlit as st
import math
import maro.cli.inspector.common_helper as common_helper
import pandas as pd
import os
import altair as alt
import numpy as np
from maro.cli.utils.params import GlobalPaths


NAME_CONVERSION_PATH = GlobalPaths.MARO_INSPECTOR_FILE_PATH['name_conversion_path']
STATIONS_FILE_PATH = GlobalPaths.MARO_INSPECTOR_FILE_PATH['stations_file_path']


def show_citi_bike_detail_plot(ROOT_PATH):
    """Show citi_bike detail plot.

    Args:
        ROOT_PATH (str): Data folder path.
    """
    dirs = os.listdir(ROOT_PATH)
    epoch_num = common_helper.get_epoch_num(len(dirs), ROOT_PATH)

    common_helper.render_H1_title('CITI_BIKE Detail Data')
    data_stations = pd.read_csv(os.path.join(ROOT_PATH, 'snapshot_0', 'stations.csv'))
    option = st.sidebar.selectbox(
        'By stations/snapshot:',
        ('by station', 'by snapshot'))
    stations_num = len(data_stations['id'].unique())
    stations_index = np.arange(stations_num).tolist()
    snapshot_num = len(data_stations['frame_index'].unique())
    snapshots_index = np.arange(snapshot_num).tolist()
    item_option_all = ['All', 'Requirement Info', 'Station Info', 'bikes',
                    'shortage', 'trip_requirement', 'fulfillment', 'capacity', 'extra_cost', 'failed_return']
    # filter by station index
    # display the change of snapshot within 1 station
    if option == 'by station':
        st.sidebar.markdown('***')
        station_index = st.sidebar.select_slider(
            'station index',
            stations_index)
        name_conversion = common_helper.read_detail_csv(CONVER_PATH)
        common_helper.render_H3_title(name_conversion.loc[int(station_index)][0] + ' Detail Data')
        # filter data by station index
        station_filtered_by_ID = data_stations[data_stations['name'] == 'stations_' + str(station_index)]
        station_sample_ratio = common_helper.holder_sample_ratio(snapshot_num)
        snapshot_sample_num = st.sidebar.select_slider('Snapshot Sampling Ratio:', station_sample_ratio)
        # get formula input & output
        data_genera = common_helper.formula_define(station_filtered_by_ID)
        if data_genera is not None:
            station_filtered_by_ID = data_genera['data_after']
            item_option_all.append(data_genera['name'])

        item_option = st.multiselect('', item_option_all, item_option_all)
        item_option = get_CITI_item_option(item_option, item_option_all)
        item_option.append("frame_index")
        station_filtered = station_filtered_by_ID[item_option].reset_index(drop=True)
        down_pooling = common_helper.get_snapshot_sample(snapshot_num, snapshot_sample_num)
        station_filtered = station_filtered.iloc[down_pooling]
        station_filtered.rename(columns={'frame_index': 'snapshot_index'}, inplace=True)
        station_filtered_long_form = station_filtered.melt('snapshot_index', var_name='Attributes',
                                                        value_name='count')
        custom_chart_station = alt.Chart(station_filtered_long_form).mark_line().encode(
            x='snapshot_index',
            y='count',
            color='Attributes',
            tooltip=['Attributes', 'count', 'snapshot_index']
        ).properties(
            width=700,
            height=380
        )
        st.altair_chart(custom_chart_station)

    # filter by snapshot index
    # display all station information within 1 snapshot
    if option == 'by snapshot':
        # get selected snapshot index
        snapshot_index = st.sidebar.select_slider(
            'snapshot index',
            snapshots_index)
        common_helper.render_H3_title('Snapshot-' + str(snapshot_index) + ':  Detail Data')
        # get according data with selected snapshot
        snapshot_filtered_by_Frame_Index = data_stations[data_stations['frame_index'] == snapshot_index]
        # get increasing rate
        sample_ratio = common_helper.holder_sample_ratio(snapshot_num)
        # get sample rate (between 0-1)
        station_sample_num = st.sidebar.select_slider('Snapshot Sampling Ratio', sample_ratio)
        # get formula input & output
        data_genera = common_helper.formula_define(snapshot_filtered_by_Frame_Index)
        if data_genera is not None:
            snapshot_filtered_by_Frame_Index = data_genera['data_after']
            item_option_all.append(data_genera['name'])
        # get selected attributes
        item_option = st.multiselect(
            '',
            item_option_all,
            item_option_all)
        # convert selected attributes into column
        item_option = get_CITI_item_option(item_option, item_option_all)
        # get sampled data & get station name
        down_pooling = list(range(0, stations_num, math.floor(1 / station_sample_num)))
        item_option.append('name')
        snapshot_filtered = snapshot_filtered_by_Frame_Index[item_option]
        snapshot_filtered = snapshot_filtered.iloc[down_pooling]
        snapshot_filtered['name'] = snapshot_filtered['name'].apply(lambda x: int(x[9:]))
        name_conversion = common_helper.read_detail_csv(os.path.join(ROOT_PATH, NAME_CONVERSION_PATH))
        snapshot_filtered['station'] = snapshot_filtered['name'].apply(lambda x: name_conversion.loc[int(x)])
        snapshot_filtered_long_form = snapshot_filtered.melt(['station', 'name'], var_name='Attributes',
                                                            value_name='count')
        custom_chart_snapshot = alt.Chart(snapshot_filtered_long_form).mark_bar().encode(
            x='name:N',
            y='count:Q',
            color='Attributes:N',
            tooltip=['Attributes', 'count', 'station'],
        ).properties(
            width=700,
            height=380
        )
        st.altair_chart(custom_chart_snapshot)


def get_CITI_item_option(item_option, item_option_all):
    """Convert selected CITI_BIKE option into column.

    Args:
        item_option(list): User selected option list.
        item_option_all(list): Pre-defined option list.

    Returns:
        list: translated users' option.

    """
    item_option_res = []
    for item in item_option:
        if item == 'All':
            item_option_all.remove('All')
            item_option_all.remove('Requirement Info')
            item_option_all.remove('Station Info')
            item_option_res = item_option_all
            break
        elif item == 'Requirement Info':
            item_option_res.append('trip_requirement')
            item_option_res.append('shortage')
            item_option_res.append('fulfillment')
        elif item == 'Station Info':
            item_option_res.append('bikes')
            item_option_res.append('extra_cost')
            item_option_res.append('failed_return')
        else:
            item_option_res.append(item)
    return item_option_res


def show_citi_bike_summary_plot(ROOT_PATH):
    """ Show summary plot.

    Args:
        ROOT_PATH (str): Data folder path.

    """
    common_helper.render_H1_title('CITI_BIKE Summary Data')
    data = common_helper.read_detail_csv(os.path.join(ROOT_PATH, STATIONS_FILE_PATH))
    name_conversion = common_helper.read_detail_csv(os.path.join(ROOT_PATH, NAME_CONVERSION_PATH))
    data['station name'] = list(map(lambda x: name_conversion.loc[int(x[9:])][0], data['name']))
    df_bikes = data[['station name', 'bikes']].sort_values(by='bikes', ascending=False).head(5)
    df_requirement = data[['station name', 'trip_requirement']].sort_values(by='trip_requirement',
                                                                        ascending=False).head(5)
    df_fulfillment = data[['station name', 'fulfillment']].sort_values(by='fulfillment', ascending=False).head(5)
    df_fulfillment_ratio = data[['station name', 'fulfillment_ratio']].sort_values(by='fulfillment_ratio',
                                                                                ascending=False).head(5)
    common_helper.generate_by_snapshot_top_summary('station name', df_bikes, 'bikes', False)
    common_helper.generate_by_snapshot_top_summary('station name', df_requirement, 'trip_requirement', False)
    common_helper.generate_by_snapshot_top_summary('station name', df_fulfillment, 'fulfillment', False)
    common_helper.generate_by_snapshot_top_summary('station name', df_fulfillment_ratio, 'fulfillment_ratio', False)


def start_citi_bike_dashboard(ROOT_PATH):
    """Entrance of citi_bike dashboard.

    Args:
        ROOT_PATH (str): Data folder path.
    """
    option = st.sidebar.selectbox(
        'Data Type',
        ('Summary', 'Detail'))
    if option == 'Summary':
        show_citi_bike_summary_plot(ROOT_PATH)
    else:
        show_citi_bike_detail_plot(ROOT_PATH)
