# usage:
# python bikedp.py ../../ny ../../ny/bin2 ../../ny/full/h3_201306_202001.station.csv

import os
import re
import sys
import csv
import math
import datetime
import numpy as np
import pandas as pd

# from geopy import distance
from collections import defaultdict

# NOTE: the order will affect the result data
input_file_list = [
    # "201306-citibike-tripdata.csv",
    # "201307-citibike-tripdata.csv",
    # "201308-citibike-tripdata.csv",
    # "201309-citibike-tripdata.csv",
    # "201310-citibike-tripdata.csv",
    # "201311-citibike-tripdata.csv",
    # "201312-citibike-tripdata.csv",
    # "201401-citibike-tripdata.csv",
    # "201402-citibike-tripdata.csv",
    # "201403-citibike-tripdata.csv",
    # "201404-citibike-tripdata.csv",
    # "201405-citibike-tripdata.csv",
    # "201406-citibike-tripdata.csv",
    # "201407-citibike-tripdata.csv",
    # "201408-citibike-tripdata.csv",
    # "201409-citibike-tripdata.csv",
    # "201410-citibike-tripdata.csv",
    # "201411-citibike-tripdata.csv",
    # "201412-citibike-tripdata.csv",
    # "201501-citibike-tripdata.csv",
    # "201502-citibike-tripdata.csv",
    # "201503-citibike-tripdata.csv",
    # "201504-citibike-tripdata.csv",
    # "201505-citibike-tripdata.csv",
    # "201506-citibike-tripdata.csv",
    # "201507-citibike-tripdata.csv",
    # "201508-citibike-tripdata.csv",
    # "201509-citibike-tripdata.csv",
    # "201510-citibike-tripdata.csv",
    # "201511-citibike-tripdata.csv",
    # "201512-citibike-tripdata.csv",
    # "201601-citibike-tripdata.csv",
    # "201602-citibike-tripdata.csv",
    # "201603-citibike-tripdata.csv",
    # "201604-citibike-tripdata.csv",
    # "201605-citibike-tripdata.csv",
    # "201606-citibike-tripdata.csv",
    # "201607-citibike-tripdata.csv",
    # "201608-citibike-tripdata.csv",
    # "201609-citibike-tripdata.csv",
    # "201610-citibike-tripdata.csv",
    # "201611-citibike-tripdata.csv",
    # "201612-citibike-tripdata.csv",
    # "201701-citibike-tripdata.csv",
    # "201702-citibike-tripdata.csv",
    # "201703-citibike-tripdata.csv",
    # "201704-citibike-tripdata.csv",
    # "201705-citibike-tripdata.csv",
    # "201706-citibike-tripdata.csv",
    # "201707-citibike-tripdata.csv",
    # "201708-citibike-tripdata.csv",
    # "201709-citibike-tripdata.csv",
    # "201710-citibike-tripdata.csv",
    # "201711-citibike-tripdata.csv",
    # "201712-citibike-tripdata.csv",
    # "201801-citibike-tripdata.csv",
    # "201802-citibike-tripdata.csv",
    # "201803-citibike-tripdata.csv",
    # "201804-citibike-tripdata.csv",
    # "201805-citibike-tripdata.csv",
    # "201806-citibike-tripdata.csv",
    # "201807-citibike-tripdata.csv",
    # "201808-citibike-tripdata.csv",
    # "201809-citibike-tripdata.csv",
    # "201810-citibike-tripdata.csv",
    # "201811-citibike-tripdata.csv",
    # "201812-citibike-tripdata.csv",
    "201901-citibike-tripdata.csv",
    "201902-citibike-tripdata.csv",
    "201903-citibike-tripdata.csv",
    "201904-citibike-tripdata.csv",
    "201905-citibike-tripdata.csv",
    "201906-citibike-tripdata.csv",
    "201907-citibike-tripdata.csv",
    "201908-citibike-tripdata.csv",
    "201909-citibike-tripdata.csv",
    "201910-citibike-tripdata.csv",
    "201911-citibike-tripdata.csv",
    "201912-citibike-tripdata.csv",
    # "202001-citibike-tripdata.csv"

    # "sample.csv"
]

usertype_map = {
    "Subscriber": 0,
    "Customer": 1
}

data_file_name = "data.bin"
mapping_file_name = "map.csv"
cell_file_name = "cell.csv"
cell_name_file_name = "cell_name.csv"


output_data_dtype = np.dtype([
    ("start_time", "datetime64[s]"),  # datetime
    ("start_station", "i2"),  # id
    ("end_station", "i2"),  # id
    ("duration", "i2"),  # min
    ("gendor", "b"),
    ("usertype", "b"),
    ("start_cell", "i2"),
    ("end_cell", "i2")
])

######### input ############


def read_src_file(file: str):
    """read and return processed rows"""
    ret = []
    stations = {}

    if os.path.exists(file):
        with open(file) as fp:

            ret = pd.read_csv(fp)
            ret = ret[['tripduration', 'starttime', 'start station id', 'end station id', 'gender', 'usertype']]
            ret['tripduration'] = pd.to_numeric(pd.to_numeric(ret['tripduration'], downcast='integer') / 60, downcast='integer')
            ret['starttime'] = pd.to_datetime(ret['starttime'])
            ret['start station id'] = pd.to_numeric(ret['start station id'], errors='coerce', downcast='integer')
            # ret['stoptime'] = pd.to_datetime(ret['stoptime'])
            ret['end station id'] = pd.to_numeric(ret['end station id'], errors='coerce', downcast='integer')
            # ret['start station latitude'] = pd.to_numeric(ret['start station latitude'],downcast='float')
            # ret['start station longitude'] = pd.to_numeric(ret['start station longitude'],downcast='float')
            # ret['end station latitude'] = pd.to_numeric(ret['end station latitude'],downcast='float')
            # ret['end station longitude'] = pd.to_numeric(ret['end station longitude'],downcast='float')
            # ret['birth year'] = pd.to_numeric(ret['birth year'],errors='coerce',downcast='integer')
            ret['gender'] = pd.to_numeric(ret['gender'], errors='coerce', downcast='integer')
            ret['usertype'] = ret['usertype'].apply(str).apply(lambda x: 0 if x in ['Subscriber', 'subscriber'] else 1 if x in ['Customer', 'customer'] else 2)
            ret.dropna(subset=['start station id', 'end station id'], inplace=True)
            ret.drop(ret[ret['tripduration'] <= 1].index, axis=0, inplace=True)
            ret = ret.sort_values(by='starttime', ascending=True)

    return ret, stations


def read_station_file(station_file_path):
    h3_data = None
    if os.path.exists(station_file_path):
        with open(station_file_path, mode="r", encoding="utf-8") as station_file:
            h3_data = pd.read_csv(station_file)
    return h3_data


def station_to_cell(station_file_path: str):
    # cell init data
    cell_data = None
    # station to cell mapping
    station_data = None
    # cell neighbors data
    mapping_map = None
    # no neighbors cells
    drop_cell = None

    if os.path.exists(station_file_path):
        with open(station_file_path, mode="r", encoding="utf-8") as station_file:
            # read station to cell file
            raw_station_data = pd.read_csv(station_file)
            # group by cell to generate cell init info
            cell_data = raw_station_data[['hex_id', 'capacity', 'init']].groupby(['hex_id']).sum().sort_values(by='hex_id', ascending=True).reset_index()
            # generate cell id by index
            cell_data['cell_id'] = pd.to_numeric(cell_data.index)
            cell_data['capacity'] = pd.to_numeric(cell_data['capacity'], downcast='integer')
            cell_data['init'] = pd.to_numeric(cell_data['init'], downcast='integer')
            # cell_data columns = ['cell_id','capacity','init','hex_id']

            # fill cell id back to station-cell mapping
            station_data = raw_station_data.join(cell_data[['cell_id', 'hex_id']].set_index('hex_id'), on='hex_id')

            print(station_data, cell_data)

            # generate cell neighbors data from column neighbors
            mapping_data = station_data[['cell_id', 'hex_id', 'neighbors']].drop_duplicates(subset=['cell_id']).reset_index()
            mapping_data['mapping'] = mapping_data['neighbors'].apply(lambda x: _gen_neighbor_mapping(x, mapping_data[['cell_id', 'hex_id']]))
            mapping_map = pd.DataFrame(-1, index=np.arange(len(mapping_data)), columns=np.arange(6), dtype=np.int64)
            mapping_data[['cell_id', 'mapping']].apply(lambda x:  _fill_mapping(x, mapping_map), axis=1)
            mapping_map['cell_id'] = mapping_data['cell_id']

            print(mapping_map)
    return cell_data, station_data, mapping_map


def _gen_neighbor_mapping(neighbors: str, neighbors_mapping: pd.DataFrame):
    # get neighbors list from neighbors string
    hex_list = re.findall(r'[0-9a-fA-F]+', neighbors)
    # remove neighbors not in cell list
    hex_df = pd.DataFrame(pd.Series(hex_list), columns=['hex_id']).join(neighbors_mapping.set_index('hex_id'), on='hex_id').reset_index()
    hex_df.dropna(subset=['cell_id'], inplace=True)
    # pick cell id of neighbors
    ret = hex_df['cell_id'].tolist()
    return ret


def _fill_mapping(row, mapping_map: pd.DataFrame):
    column = 0
    # loop neighbors list column in row
    for i in range(len(row['mapping'])):
        x = row['cell_id']
        # filter self in neighbors
        if row['mapping'][i] != x:
            # fill mapping_map
            mapping_map.loc[x, column] = int(row['mapping'][i])
            # skip self, use new column c in mapping_map 
            column += 1



def _filter_mapping(row, cell_list):
    for cell in row:
        if cell not in cell_list:
            cell = -1
    return row

######### output ############

def concat(data: pd.DataFrame, file: str, station_data: pd.DataFrame, mapping_map: pd.DataFrame):
    ret = data[['starttime', 'start station id', 'end station id', 'tripduration', 'usertype', 'gender']]

    # get the file size
    file_size = 0
    ret = ret.join(station_data[['station_id', 'cell_id']].set_index('station_id'), on='start station id').rename(columns={'cell_id': 'start_cell'})
    ret = ret.join(station_data[['station_id', 'cell_id']].set_index('station_id'), on='end station id').rename(columns={'cell_id': 'end_cell'})
    ret = ret.rename(columns={'starttime': 'start_time', 'start station id': 'start_station', 'end station id': 'end_station', 'tripduration': 'duration'})
    ret = ret[['start_time', 'start_station', 'end_station', 'duration', 'gender', 'usertype', 'start_cell', 'end_cell']]

    # generate cell need to be dropped, because it has no neighbors
    used_cells = []
    used_cells.append(ret[['start_cell']].drop_duplicates(subset=['start_cell']).rename(columns={'start_cell': 'cell_id'}))
    used_cells.append(ret[['end_cell']].drop_duplicates(subset=['end_cell']).rename(columns={'end_cell': 'cell_id'}))
    data_cell = pd.concat(used_cells).drop_duplicates().sort_values(by=['cell_id']).reset_index()
    data_mapping_data = mapping_data.set_index('cell_id')
    data_mapping_data = data_cell.join(data_mapping_data, on='cell_id').set_index('cell_id').drop(['index'], axis = 1)
    data_mapping_data = data_mapping_data.apply(lambda x: _filter_mapping(x,data_cell), axis = 1)

    print(data_mapping_data)
    drop_cell = pd.to_numeric(data_mapping_data[data_mapping_data.sum(axis=1) == -6].index).to_list()
    print(drop_cell)
    # drop cell have no neighbors
    ret.drop(ret[ret['start_cell'].apply(lambda x: x in drop_cell) | ret['end_cell'] .apply(lambda x: x in drop_cell)].index, axis=0, inplace=True)

    mem_output = list(ret.itertuples(index=False, name=None))

    if not os.path.exists(file):
        with open(file, "w+") as fp:
            pass

    with open(file, "rb") as fp:
        fp.seek(0, 2)
        file_size = fp.tell()

    # append to the end
    item_num = len(mem_output)
    arr = np.memmap(file, dtype=output_data_dtype, offset=file_size, shape=(item_num, ))

    arr[:] = np.array(mem_output, dtype=output_data_dtype)

    
    return data_cell



    


if __name__ == "__main__":
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    station_file_path = sys.argv[3]

    output_data_path = os.path.join(output_folder, data_file_name)
    cell_file_path = os.path.join(output_folder, cell_file_name)
    mapping_file_path = os.path.join(output_folder, mapping_file_name)
    cell_name_file_path = os.path.join(output_folder, cell_name_file_name)

    # generate full cell data
    cell_init, station_data, mapping_data = station_to_cell(station_file_path)

    data_cell_dfs = []
    # generate bin file
    for src_file in input_file_list:
        src_full_path = os.path.join(input_folder, src_file)

        # show current process file
        print(src_full_path)

        r, s = read_src_file(src_full_path)

        if r is not None:
            data_cell_dfs.append(concat(r, output_data_path, station_data, mapping_data))

    # filter cell by data
    data_cell = pd.concat(data_cell_dfs).drop_duplicates(subset=['cell_id']).sort_values(by=['cell_id']).reset_index()
    data_cell = data_cell[['cell_id']]
    print(data_cell)
    data_cell_init = data_cell.join(cell_init.set_index('cell_id'), on='cell_id')

    data_cell_name = data_cell_init[['cell_id','hex_id']]
    data_cell_init = data_cell_init[['cell_id', 'capacity', 'init']]

    data_mapping_data = data_cell_init[['cell_id']].join(mapping_data.set_index('cell_id'), on='cell_id')
    data_mapping_data= data_mapping_data.apply(lambda x: _filter_mapping(x,data_cell_init['cell_id']), axis = 1)


    # write cell init file
    with open(cell_file_path, mode="w", encoding="utf-8", newline='') as cell_file:
        data_cell_init.to_csv(cell_file, index=False)

    # write cell neighbors file
    with open(mapping_file_path, mode="w", encoding="utf-8", newline='') as mapping_file:
        data_mapping_data.to_csv(mapping_file, index=False)

    # write cell name file
    with open(cell_name_file_path, mode="w", encoding="utf-8", newline='') as cell_name_file:
        data_cell_name.to_csv(cell_name_file, index=False)