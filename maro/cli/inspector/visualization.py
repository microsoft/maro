import os
import math
import argparse
import streamlit as st
import altair as alt
import numpy as np
import pandas as pd

Title_html = """
<style>
    .title h1{
        font-size: 30px;
        color: black;
        background-size: 600vw 600vw;
        animation: slide 10s linear infinite forwards;
        margin-left:100px;
        text-align:left;
        margin-left:50px;
    }
    .title h3{
        font-size: 20px;
        color: grey;
        background-size: 600vw 600vw;
        animation: slide 10s linear infinite forwards;
        text-align:center
    }
    @keyframes slide {
        0%{
        background-position-x: 0%;
        }
        100%{
        background-position-x: 600vw;
        }
    }
</style>
"""


def generate_top_summary(data, snapshot_index, ports_num, CONVER_PATH):
    data_acc = data[data['frame_index'] == snapshot_index].reset_index(drop=True)
    data_acc['fulfillment_ratio'] = list(
        map(lambda x, y: float('{:.4f}'.format(x / (y + 1 / 1000))), data_acc['acc_fulfillment'],
            data_acc['acc_booking']))
    data_acc['name'] = data_acc['name'].apply(lambda x: int(x[6:]))
    # data_acc.rename(columns={'name': 'port name'}, inplace=True)
    name_conversion = read_name_conversion(CONVER_PATH)
    data_acc['port name'] = data_acc['name'].apply(lambda x: name_conversion.loc[int(x)])
    df_booking = data_acc[['port name', 'acc_booking']].sort_values(by='acc_booking', ascending=False).head(5)
    df_fulfillment = data_acc[['port name', 'acc_fulfillment']].sort_values(by='acc_fulfillment',
                                                                            ascending=False).head(5)
    df_shortage = data_acc[['port name', 'acc_shortage']].sort_values(by='acc_shortage', ascending=False).head(5)
    df_ratio = data_acc[['port name', 'fulfillment_ratio']].sort_values(by='fulfillment_ratio', ascending=False).head(5)
    generate_by_snapshot_top_summary('port name', df_booking, 'acc_booking', True, snapshot_index)
    generate_by_snapshot_top_summary('port name', df_fulfillment, 'acc_fulfillment', True, snapshot_index)
    generate_by_snapshot_top_summary('port name', df_shortage, 'acc_shortage', True, snapshot_index)
    generate_by_snapshot_top_summary('port name', df_ratio, 'fulfillment_ratio', True, snapshot_index)


def generate_by_snapshot_top_summary(attr_name, data, attribute, Need_SnapShot, snapshot_index=-1):
    if Need_SnapShot:
        render_H3_title('SnapShot-' + str(snapshot_index) + ': ' + '     Top 5 ' + attribute)
    else:
        render_H3_title('Top 5 ' + attribute)
    data['counter'] = range(len(data))
    data[attr_name] = list(map(lambda x, y: str(x + 1) + '-' + y, data['counter'], data[attr_name]))
    bars = alt.Chart(data).mark_bar().encode(
        x=attribute + ':Q',
        y=attr_name + ":O",
    ).properties(
        width=700,
        height=240
    )
    text = bars.mark_text(
        align='left',
        baseline='middle',
        dx=3
    ).encode(
        text=attribute + ':Q'
    )
    st.altair_chart(bars + text)


def generate_detail_vessel_by_snapshot(data_vessels, snapshot_index, vessels_num):
    render_H3_title('SnapShot-' + str(snapshot_index) + ': Vessel Attributes')
    sample_ratio = holder_sample_ratio(vessels_num)
    sample_ratio_res = st.sidebar.select_slider('Vessels Sample Ratio:', sample_ratio)
    down_pooling = list(range(0, vessels_num, math.floor(1 / sample_ratio_res)))
    snapshot_filtered = data_vessels[data_vessels['frame_index'] == snapshot_index].reset_index(drop=True)
    snapshot_filtered = snapshot_filtered[['capacity', 'full', 'empty', 'remaining_space', 'name']]
    snapshot_temp = pd.DataFrame()
    for index in down_pooling:
        snapshot_temp = pd.concat(
            [snapshot_temp, snapshot_filtered[snapshot_filtered['name'] == 'vessels_' + str(index)]], axis=0)
    snapshot_filtered = snapshot_temp.reset_index(drop=True)
    snapshot_filtered['name'] = snapshot_filtered['name'].apply(lambda x: int(x[8:]))
    snapshot_filtered_long_form = snapshot_filtered.melt('name', var_name='Attributes', value_name='count')
    custom_chart_snapshot = alt.Chart(snapshot_filtered_long_form).mark_bar().encode(
        x=alt.X('name:N', axis=alt.Axis(title='vessel name')),
        y='count:Q',
        color='Attributes:N',
        tooltip=['Attributes', 'count', 'name']
    ).properties(
        width=700,
        height=380
    )
    st.altair_chart(custom_chart_snapshot)


# generate detail plot (comprehensive,detail,specific)
# view info within different snapshot in the same epoch
def generate_detail_plot_by_snapshot(info_selector, data, snapshot_index, ports_num,
                                    CONVER_PATH, sample_ratio_res, item_option=None):
    data_acc = data[info_selector]
    # delete parameter:frame_index
    info_selector.pop(1)
    down_pooling = list(range(0, ports_num, math.floor(1 / sample_ratio_res)))
    snapshot_filtered = data_acc[data_acc['frame_index'] == snapshot_index][info_selector].reset_index(drop=True)
    snapshot_temp = pd.DataFrame(columns=info_selector)
    for index in down_pooling:
        snapshot_temp = pd.concat(
            [snapshot_temp, snapshot_filtered[snapshot_filtered['name'] == 'ports_' + str(index)]], axis=0)
    snapshot_temp['name'] = snapshot_temp['name'].apply(lambda x: int(x[6:]))
    snapshot_filtered = snapshot_temp.reset_index(drop=True)
    if item_option is not None:
        item_option.append('name')
        snapshot_filtered = snapshot_filtered[item_option]

    name_conversion = read_name_conversion(CONVER_PATH)
    snapshot_filtered['port name'] = snapshot_filtered['name'].apply(lambda x: name_conversion.loc[int(x)])
    snapshot_filtered_long_form = snapshot_filtered.melt(['name', 'port name'], var_name='Attributes',
                                                        value_name='count')
    custom_chart_snapshot = alt.Chart(snapshot_filtered_long_form).mark_bar().encode(
        x='name:N',
        y='count:Q',
        color='Attributes:N',
        tooltip=['Attributes', 'count', 'port name']
    ).properties(
        width=700,
        height=380
    )
    st.altair_chart(custom_chart_snapshot)


# generate detail plot (comprehensive,detail,specific)
# view info within different holders(ports,stations,etc) in the same epoch
# change snapshot sampling num freely
def generate_detail_plot_by_ports(info_selector, data, str_temp, snapshot_num, snapshot_sample_num, item_option=None):
    data_acc = data[info_selector]
    # delete parameter:name
    info_selector.pop(0)
    down_pooling = list(range(1, snapshot_num, math.floor(1 / snapshot_sample_num)))
    down_pooling.insert(0, 0)
    if snapshot_num - 1 not in down_pooling:
        down_pooling.append(snapshot_num - 1)
    port_filtered = data_acc[data_acc['name'] == str_temp][info_selector].reset_index(drop=True)
    port_filtered.rename(columns={'frame_index': 'snapshot_index'}, inplace=True)

    bar_data = port_filtered.loc[down_pooling]
    if item_option is not None:
        item_option.append('snapshot_index')
        bar_data = bar_data[item_option]
    bar_data_long_form = bar_data.melt('snapshot_index', var_name='Attributes', value_name='count')
    custom_chart_port = alt.Chart(bar_data_long_form).mark_line().encode(
        x='snapshot_index',
        y='count',
        color='Attributes',
        tooltip=['Attributes', 'count', 'snapshot_index']
    ).properties(
        width=700,
        height=380
    )
    st.altair_chart(custom_chart_port)

    custom_bar_chart = alt.Chart(bar_data_long_form).mark_bar().encode(
        x='snapshot_index:N',
        y='count:Q',
        color='Attributes:N',
        tooltip=['Attributes', 'count', 'snapshot_index']
    ).properties(width=700,
                height=380)
    st.altair_chart(custom_bar_chart)


@st.cache(allow_output_mutation=True)
def read_detail_csv(path):
    data = pd.read_csv(path)
    return data


def holder_sample_ratio(snapshot_num):
    snapshot_sample_origin = round(1 / snapshot_num, 4)
    snapshot_sample_ratio = np.arange(snapshot_sample_origin, 1, snapshot_sample_origin).tolist()
    sample_ratio = [float('{:.4f}'.format(i)) for i in snapshot_sample_ratio]
    if 1 not in sample_ratio:
        sample_ratio.append(1)
    return sample_ratio


def generate_hot_map(matrix_data):
    matrix_data = matrix_data.replace('[', '')
    matrix_data = matrix_data.replace(']', '')
    matrix_data = matrix_data.split()
    matrix_len = int(math.sqrt(len(matrix_data)))
    b = np.array(matrix_data).reshape(matrix_len, matrix_len)
    x_axis_single = list(range(0, matrix_len))
    x_axis = [x_axis_single] * matrix_len
    y_axis = [[row[col] for row in x_axis] for col in range(len(x_axis[0]))]
    # Convert this grid to columnar data expected by Altair
    source = pd.DataFrame({'dest_port': np.array(x_axis).ravel(),
                        'start_port': np.array(y_axis).ravel(),
                        'count': np.array(b).ravel()})
    chart = alt.Chart(source).mark_rect().encode(
        x='dest_port:O',
        y='start_port:O',
        color='count:Q',
        tooltip=['dest_port', 'start_port', 'count']
    ).properties(
        width=700,
        height=380
    )
    st.altair_chart(chart)


def show_volume_hot_map(ROOT_PATH, senario, epoch_index, snapshot_index):
    matrix_data = pd.read_csv(os.path.join(ROOT_PATH, 'snapshot_' + str(epoch_index), 'matrices.csv')).loc[
        snapshot_index]
    if senario == 'CIM':
        render_H3_title('SnapShot-' + str(snapshot_index) + ': Acc Port Transfer Volume')
        generate_hot_map(matrix_data['full_on_ports'])


# Convert selected CITI_BIKE option into column
def get_CITI_item_option(item_option, item_option_all):
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


# Convert selected CIM option into column
def get_CIM_item_option(item_option, item_option_all):
    item_option_res = []
    for item in item_option:
        if item == 'All':
            item_option_all.remove('All')
            item_option_all.remove('Booking Info')
            item_option_all.remove('Port Info')
            item_option_res = item_option_all
            break
        elif item == 'Booking Info':
            item_option_res.append('booking')
            item_option_res.append('shortage')
            item_option_res.append('fulfillment')
        elif item == 'Port Info':
            item_option_res.append('full')
            item_option_res.append('empty')
            item_option_res.append('on_shipper')
            item_option_res.append('on_consignee')
        else:
            item_option_res.append(item)
    return item_option_res


def render_H1_title(content):
    html_title = Title_html + '<div class="title"><h1>' + content + '</h1></div>'
    st.markdown(html_title, unsafe_allow_html=True)


def render_H3_title(content):
    html_title = Title_html + '<div class="title"><h3>' + content + '</h3></div>'
    st.markdown(html_title, unsafe_allow_html=True)


# entrance of detail plot
def show_detail_plot(senario, ROOT_PATH, CONVER_PATH):
    dirs = os.listdir(ROOT_PATH)
    epoch_num = get_epoch_num(len(dirs), ROOT_PATH)

    if senario == 'CIM':
        option_epoch = st.sidebar.select_slider(
            'Choose an Epoch:',
            list(range(0, epoch_num)))
        dir = os.path.join(ROOT_PATH, 'snapshot_' + str(option_epoch))
        data_ports = read_detail_csv(os.path.join(dir, 'ports.csv'))
        data_ports['remaining_space'] = list(map(lambda x, y, z: x - y - z, data_ports['capacity'],
                                                data_ports['full'], data_ports['empty']))
        ports_num = len(data_ports['name'].unique())
        ports_index = np.arange(ports_num).tolist()
        snapshot_num = len(data_ports['frame_index'].unique())
        snapshots_index = np.arange(snapshot_num).tolist()

        st.sidebar.markdown('***')
        option_2 = st.sidebar.selectbox(
            'By ports/snapshot:',
            ('by ports', 'by snapshot'))
        comprehensive_info = ['name', 'frame_index', 'acc_shortage', 'acc_booking', 'acc_fulfillment']
        specific_info = ['name', 'frame_index', 'shortage', 'booking', 'fulfillment', 'on_shipper', 'on_consignee',
                        'capacity', 'full', 'empty', 'remaining_space']
        item_option_all = ['All', 'Booking Info', 'Port Info', 'shortage', 'booking', 'fulfillment', 'on_shipper',
                        'on_consignee', 'capacity', 'full', 'empty', 'remaining_space']
        if option_2 == 'by ports':
            port_index = st.sidebar.select_slider(
                'Choose a Port:',
                ports_index)
            str_port_option = 'ports_' + str(port_index)
            name_conversion = read_name_conversion(CONVER_PATH)
            sample_ratio = holder_sample_ratio(snapshot_num)
            snapshot_sample_num = st.sidebar.select_slider('Snapshot Sampling Ratio:', sample_ratio)

            render_H1_title('CIM Acc Data')
            render_H3_title('Port Acc Attributes: ' + str(port_index) + ' -  ' +
                            name_conversion.loc[int(port_index)][0])
            generate_detail_plot_by_ports(comprehensive_info, data_ports, str_port_option,
                                        snapshot_num, snapshot_sample_num)
            render_H1_title('CIM Detail Data')
            data_genera = formula_define(data_ports)
            if data_genera is not None:
                specific_info.append(data_genera['name'])
                data_ports = data_genera['data_after']
                item_option_all.append(data_genera['name'])
            item_option = st.multiselect(
                ' ', item_option_all, item_option_all)
            item_option = get_CIM_item_option(item_option, item_option_all)
            str_temp = 'ports_' + str(port_index)
            render_H3_title(
                'Port Detail Attributes: ' + str(port_index) + ' -  ' + name_conversion.loc[int(port_index)][0])
            generate_detail_plot_by_ports(specific_info, data_ports, str_temp, snapshot_num,
                                        snapshot_sample_num, item_option)
        if option_2 == 'by snapshot':
            snapshot_index = st.sidebar.select_slider(
                'snapshot index',
                snapshots_index)
            sample_ratio = holder_sample_ratio(ports_num)
            sample_ratio_res = st.sidebar.select_slider('Ports Sample Ratio:', sample_ratio)
            render_H1_title('Acc Data')
            show_volume_hot_map(ROOT_PATH, senario, option_epoch, snapshot_index)
            render_H3_title('SnapShot-' + str(snapshot_index) + ': Port Acc Attributes')
            generate_detail_plot_by_snapshot(comprehensive_info, data_ports, snapshot_index, ports_num,
                                            CONVER_PATH, sample_ratio_res)
            generate_top_summary(data_ports, snapshot_index, ports_num, CONVER_PATH)
            render_H1_title('Detail Data')
            data_vessels = read_detail_csv(os.path.join(dir, 'vessels.csv'))
            vessels_num = len(data_vessels['name'].unique())
            generate_detail_vessel_by_snapshot(data_vessels, snapshot_index, vessels_num)
            render_H3_title('SnapShot-' + str(snapshot_index) + ': Port Detail Attributes')
            data_genera = formula_define(data_ports)
            if data_genera is not None:
                specific_info.append(data_genera['name'])
                data_ports = data_genera['data_after']
                item_option_all.append(data_genera['name'])
            item_option = st.multiselect(' ', item_option_all, item_option_all)
            item_option = get_CIM_item_option(item_option, item_option_all)
            generate_detail_plot_by_snapshot(specific_info, data_ports, snapshot_index, ports_num,
                                            CONVER_PATH, sample_ratio_res, item_option)

    elif senario=='CITI_BIKE':
        render_H1_title(senario + ' Detail Data')
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
            name_conversion = read_name_conversion(CONVER_PATH)
            render_H3_title(name_conversion.loc[int(station_index)][0] + ' Detail Data')
            station_filtered_by_ID = data_stations[data_stations['name'] == 'stations_' + str(station_index)]
            station_sample_ratio = holder_sample_ratio(snapshot_num)
            snapshot_sample_num = st.sidebar.select_slider('Snapshot Sampling Ratio:', station_sample_ratio)

            data_genera = formula_define(station_filtered_by_ID)
            if data_genera is not None:
                station_filtered_by_ID = data_genera['data_after']
                item_option_all.append(data_genera['name'])

            item_option = st.multiselect('', item_option_all, item_option_all)
            item_option = get_CITI_item_option(item_option, item_option_all)
            item_option.append("frame_index")
            station_filtered = station_filtered_by_ID[item_option].reset_index(drop=True)
            down_pooling = list(range(1, snapshot_num, math.floor(1 / snapshot_sample_num)))
            down_pooling.insert(0, 0)
            if snapshot_num - 1 not in down_pooling:
                down_pooling.append(snapshot_num - 1)
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
            snapshot_index = st.sidebar.select_slider(
                'snapshot index',
                snapshots_index)
            render_H3_title('Snapshot-' + str(snapshot_index) + ':  Detail Data')
            snapshot_filtered_by_Frame_Index = data_stations[data_stations['frame_index'] == snapshot_index]
            sample_ratio = holder_sample_ratio(snapshot_num)
            station_sample_num = st.sidebar.select_slider('Snapshot Sampling Ratio', sample_ratio)
            data_genera = formula_define(snapshot_filtered_by_Frame_Index)
            if data_genera is not None:
                snapshot_filtered_by_Frame_Index = data_genera['data_after']
                item_option_all.append(data_genera['name'])
            item_option = st.multiselect(
                '',
                item_option_all,
                item_option_all)
            item_option = get_CITI_item_option(item_option, item_option_all)
            down_pooling = list(range(0, stations_num, math.floor(1 / station_sample_num)))
            item_option.append('name')
            snapshot_filtered = snapshot_filtered_by_Frame_Index[item_option]
            snapshot_temp = pd.DataFrame(columns=item_option)
            for index in down_pooling:
                snapshot_temp = pd.concat(
                    [snapshot_temp, snapshot_filtered[snapshot_filtered['name'] == 'stations_' + str(index)]], axis=0)

            snapshot_filtered = snapshot_temp
            snapshot_filtered['name'] = snapshot_filtered['name'].apply(lambda x: int(x[9:]))
            name_conversion = read_name_conversion(CONVER_PATH)
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


@st.cache
def read_name_conversion(path):
    return pd.read_csv(path)


def formula_define(data_origin):
    st.sidebar.markdown('***')
    formula_select = st.sidebar.selectbox('formula:', ['a+b', 'a*b+sqrt(c*d)'])
    paras = st.sidebar.text_input('parameters separated by ;')
    res = paras.split(';')
    if formula_select == 'a+b':
        if len(res) == 0 or res[0] == "":
            return
        elif len(res) != 2:
            st.warning('input parameter number wrong')
            return
        else:
            data_right = judge_append_data(data_origin.head(0), res)
            if data_right:
                data_origin[res[0] + '+' + res[1]] = list(
                    map(lambda x, y: x + y, data_origin[res[0]], data_origin[res[1]]))
            else:
                return
    if formula_select == 'a*b+sqrt(c*d)':
        if len(res) == 0 or res[0] == "":
            return
        elif len(res) != 4:
            st.warning('input parameter number wrong')
            return
        else:
            data_right = judge_append_data(data_origin.head(0), res)
            if data_right:
                data_origin[res[0] + '*' + res[1] + '+sqrt(' + res[2] + '*+' + res[3] + ')'] = list(
                    map(lambda x, y, z, w: int(x) * int(y) + math.sqrt(z * int(w)),
                        data_origin[res[0]], data_origin[res[1]], data_origin[res[2]], data_origin[res[3]]))
            else:
                return
    data = {'data_after': data_origin, 'name': res[0] + '*' + res[1] + '+sqrt(' + res[2] + '*+' + res[3] + ')'}
    return data


def judge_append_data(data_head, res):
    data_right = True
    for item in res:
        if item not in data_head:
            data_right = False
            st.warning('parameter name:' + item + ' not exist')
    return data_right


@st.cache
def read_single_csv(input_path):
    df_chunk = pd.read_csv(input_path, chunksize=1000)
    res_chunk = []
    for chunk in df_chunk:
        res_chunk.append(chunk)
    res_df = pd.concat(res_chunk)
    return res_df


@st.cache(allow_output_mutation=False)
def read_summary_data(file_path, down_pooling_range):
    data_epochs_all = pd.read_csv(file_path)
    data_epochs = data_epochs_all.iloc[down_pooling_range]
    return data_epochs


# generate summary plot
# view info within different epochs
def generate_summary_plot(item_option, data, down_pooling_range):
    data['epoch index'] = list(down_pooling_range)
    data_long_form = data.melt('epoch index', var_name='Attributes',
                            value_name='count')
    custom_chart_port = alt.Chart(data_long_form).mark_line().encode(
        x='epoch index',
        y='count',
        color='Attributes',
        tooltip=['Attributes', 'count', 'epoch index']
    ).properties(
        width=700,
        height=380
    )
    st.altair_chart(custom_chart_port)

    custom_chart_port_bar = alt.Chart(data_long_form).mark_bar().encode(
        x='epoch index:N',
        y='count:Q',
        color='Attributes:N',
        tooltip=['Attributes', 'count', 'epoch index']
    ).properties(
        width=700,
        height=380
    )
    st.altair_chart(custom_chart_port_bar)


# get epoch num by counting folders start with 'snapshot_'
def get_epoch_num(origin_len, ROOT_PATH):
    epoch_num = 0
    for index in range(0, origin_len):
        if os.path.exists(os.path.join(ROOT_PATH, r'snapshot_' + str(index))):
            epoch_num = epoch_num + 1
    return epoch_num


# entrance of summary plot
def show_summary_plot(senario, ROOT_PATH, CONVER_PATH, ports_file_path, stations_file_path):
    render_H1_title(senario + ' Summary Data')
    if senario == 'CIM':
        dirs = os.listdir(ROOT_PATH)
        epoch_num = get_epoch_num(len(dirs), ROOT_PATH)
        sample_ratio = holder_sample_ratio(epoch_num)
        start_epoch = st.sidebar.number_input('Start Epoch', 0, epoch_num - 1, 0)
        end_epoch = st.sidebar.number_input('End Epoch', 0, epoch_num - 1, epoch_num - 1)

        down_pooling_num = st.sidebar.select_slider(
            'Epoch Sampling Ratio',
            sample_ratio)
        down_pooling_len = math.floor(1 / down_pooling_num)
        down_pooling_range = generate_down_pooling_sample(epoch_num, down_pooling_len, start_epoch, end_epoch)
        item_option_all = ['All', 'Booking Info', 'Port Info', 'shortage', 'booking', 'fulfillment', 'on_shipper',
                        'on_consignee', 'capacity', 'full', 'empty']
        data = read_summary_data(ports_file_path, down_pooling_range)
        data_genera = formula_define(data)
        if data_genera is not None:
            data = data_genera['data_after']
            item_option_all.append(data_genera['name'])
        item_option = st.multiselect(
            ' ', item_option_all, item_option_all)
        item_option = get_CIM_item_option(item_option, item_option_all)
        data = data[item_option]
        generate_summary_plot(item_option, data, down_pooling_range)
    else:
        data = pd.read_csv(stations_file_path)
        name_conversion = read_name_conversion(CONVER_PATH)
        data['station name'] = list(map(lambda x: name_conversion[int(x[9:])], data['name']))
        df_bikes = data[['station name', 'bikes']].sort_values(by='bikes', ascending=False).head(5)
        df_requirement = data[['station name', 'trip_requirement']].sort_values(by='trip_requirement',
                                                                            ascending=False).head(5)
        df_fulfillment = data[['station name', 'fulfillment']].sort_values(by='fulfillment', ascending=False).head(5)
        df_fulfillment_ratio = data[['station name', 'fulfillment_ratio']].sort_values(by='fulfillment_ratio',
                                                                                    ascending=False).head(5)
        generate_by_snapshot_top_summary('station name', df_bikes, 'bikes', False)
        generate_by_snapshot_top_summary('station name', df_requirement, 'trip_requirement', False)
        generate_by_snapshot_top_summary('station name', df_fulfillment, 'fulfillment', False)
        generate_by_snapshot_top_summary('station name', df_fulfillment_ratio, 'fulfillment_ratio', False)


# generate down pooling list based on origin data and down pooling rate
def generate_down_pooling_sample(origin_len, down_pooling_len, start_epoch, end_epoch):
    down_pooling_range = list(range(start_epoch, end_epoch, down_pooling_len))
    if end_epoch not in down_pooling_range:
        down_pooling_range.append(end_epoch)
    return down_pooling_range


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("rootpath")
    args = parser.parse_args()

    ROOT_PATH = args.rootpath

    # path to restore summary files
    ports_file_path = os.path.join(ROOT_PATH, r'snapshot_ports_summary.csv')
    vessels_file_path = os.path.join(ROOT_PATH, r'snapshot_vessels_summary.csv')
    stations_file_path = os.path.join(ROOT_PATH, r'snapshot_stations_summary.csv')
    name_conversion_path = os.path.join(ROOT_PATH, r'name_conversion.csv')

    if os.path.exists(os.path.join(ROOT_PATH, 'snapshot_0', 'ports.csv')):
        senario = 'CIM'
    else:
        senario = 'CITI_BIKE'

    if senario == 'CIM':
        option = st.sidebar.selectbox(
            'Data Type',
            ('Extro Epoch', 'Intra Epoch'))
        if option == 'Extro Epoch':
            show_summary_plot(senario, ROOT_PATH, name_conversion_path, ports_file_path, stations_file_path)
        else:
            show_detail_plot(senario, ROOT_PATH, name_conversion_path)
    elif senario == 'CITI_BIKE':
        option = st.sidebar.selectbox(
            'Data Type',
            ('Summary', 'Detail'))
        if option == 'Summary':
            show_summary_plot(senario, ROOT_PATH, name_conversion_path, ports_file_path, stations_file_path)
        else:
            show_detail_plot(senario, ROOT_PATH, name_conversion_path)
