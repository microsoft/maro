# usage:
# python bikedp.py ../../ny ../../ny/bin2019vStation ../../ny/full/h3_201306_202001.station.csv ../../ny/bin2019vStation/cell.csv

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

from h3 import h3

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
neighbor_file_name = "map.csv"
cell_init_file_name = "cell.csv"
cell_to_hex_file_name = "cell_name.csv"


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

    if os.path.exists(file):
        with open(file) as fp:

            ret = pd.read_csv(fp)
            ret = ret[['tripduration', 'starttime', 'start station id', 'end station id', 'start station latitude', 'start station longitude', 'end station latitude', 'end station longitude', 'gender', 'usertype']]
            ret['tripduration'] = pd.to_numeric(pd.to_numeric(ret['tripduration'], downcast='integer') / 60, downcast='integer')
            ret['starttime'] = pd.to_datetime(ret['starttime'])
            ret['start station id'] = pd.to_numeric(ret['start station id'], errors='coerce', downcast='integer')
            # ret['stoptime'] = pd.to_datetime(ret['stoptime'])
            ret['end station id'] = pd.to_numeric(ret['end station id'], errors='coerce', downcast='integer')
            ret['start station latitude'] = pd.to_numeric(ret['start station latitude'], downcast='float')
            ret['start station longitude'] = pd.to_numeric(ret['start station longitude'], downcast='float')
            ret['end station latitude'] = pd.to_numeric(ret['end station latitude'], downcast='float')
            ret['end station longitude'] = pd.to_numeric(ret['end station longitude'], downcast='float')
            # ret['birth year'] = pd.to_numeric(ret['birth year'],errors='coerce',downcast='integer')
            ret['gender'] = pd.to_numeric(ret['gender'], errors='coerce', downcast='integer')
            ret['usertype'] = ret['usertype'].apply(str).apply(lambda x: 0 if x in ['Subscriber', 'subscriber'] else 1 if x in ['Customer', 'customer'] else 2)
            ret.dropna(subset=['start station id', 'end station id'], inplace=True)
            ret.drop(ret[ret['tripduration'] <= 1].index, axis=0, inplace=True)
            ret = ret.sort_values(by='starttime', ascending=True)

    return ret


def _read_exist_cells(file: str):
    # read exist cells from file
    ret = pd.DataFrame(columns=['cell_id', 'hex_id'])

    if os.path.exists(file):
        with open(file) as fp:
            exist_cells = pd.read_csv(fp)
            exist_cells['cell_id'] = pd.to_numeric(exist_cells['cell_id'], downcast='integer')
            ret = ret.append(exist_cells, ignore_index=True)
    return ret


def _read_cell_init(station_file_path: str):
    # read cell init data from file
    cell_init = None

    if os.path.exists(station_file_path):
        with open(station_file_path, mode="r", encoding="utf-8") as station_file:
            # read station to cell file
            raw_station_data = pd.read_csv(station_file)
            # group by cell to generate cell init info
            cell_init = raw_station_data[['station_id', 'capacity', 'init']].reset_index()
            # generate cell id by index
            cell_init['capacity'] = pd.to_numeric(cell_init['capacity'], downcast='integer')
            cell_init['init'] = pd.to_numeric(cell_init['init'], downcast='integer')
            # cell_data columns = ['cell_id','capacity','init','hex_id']

    return cell_init


def _find_neighbors_by_row(row, cell_to_hex, tar_num):
    neighbors = []
    cur_distance = 1
    # check if there are 6 neighbors in 100 cell distance
    while len(neighbors) < tar_num and cur_distance < 100:
        new_neighbors = pd.Series(list(h3.k_ring_distances(row['hex_id'], cur_distance)[cur_distance]))
        selected_neighbors = new_neighbors[new_neighbors.isin(cell_to_hex['hex_id'])].to_list()
        neighbors = (neighbors + selected_neighbors)[:tar_num]
        cur_distance += 1
        if cur_distance == 100:
            print(row)
    return neighbors


def _gen_neighbor_mapping(neighbors: str, cell_to_hex: pd.DataFrame):
    # get neighbors list from neighbors string
    hex_list = re.findall(r'[0-9a-fA-F]+', neighbors)
    # remove neighbors not in cell list
    hex_df = pd.DataFrame(pd.Series(hex_list), columns=['hex_id']).join(cell_to_hex.set_index('hex_id'), on='hex_id').reset_index()
    hex_df.dropna(subset=['cell_id'], inplace=True)
    # pick cell id of neighbors
    ret = hex_df['cell_id'].tolist()
    return ret


def _gen_neighbor_mapping_v2(neighbors: list, cell_to_hex: pd.DataFrame):
    # remove neighbors not in cell list
    hex_df = pd.DataFrame()
    hex_df['hex_id'] = pd.Series(neighbors)
    hex_df = hex_df.join(cell_to_hex[['cell_id', 'hex_id']].set_index('hex_id'), on='hex_id')
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
            mapping_map.loc[mapping_map.index == x, column] = int(row['mapping'][i])
            # skip self, use new column c in mapping_map
            column += 1

######### output ############


def concat(data: pd.DataFrame, file: str, cells_existed: pd.DataFrame):
    ret = data[['starttime', 'start station id', 'end station id', 'tripduration', 'usertype', 'gender', 'start station latitude', 'start station longitude', 'end station latitude', 'end station longitude']]

    # get the file size
    file_size = 0
    ret['start_cell_hex'] = ret['start station id']
    ret['end_cell_hex'] = ret['end station id']

    # get new cells
    used_cells = []
    used_cells.append(ret[['start_cell_hex']].drop_duplicates(subset=['start_cell_hex']).rename(columns={'start_cell_hex': 'hex_id'})[['hex_id']])
    used_cells.append(ret[['end_cell_hex']].drop_duplicates(subset=['end_cell_hex']).rename(columns={'end_cell_hex': 'hex_id'})[['hex_id']])
    in_data_cell = pd.concat(used_cells, ignore_index=True).drop_duplicates(subset=['hex_id']).sort_values(by=['hex_id']).reset_index()[['hex_id']]
    new_cell = (in_data_cell[~in_data_cell['hex_id'].isin(cells_existed['hex_id'])].reset_index())[['hex_id']]
    max_cell_id = cells_existed['cell_id'].max() if len(cells_existed) > 0 else -1
    new_cell['cell_id'] = pd.to_numeric(new_cell.index, downcast='integer') + max_cell_id + 1
    # append the new cells to existed cells and get new cell ids
    cells_existed = cells_existed[['cell_id', 'hex_id']].append(new_cell, ignore_index=True)
    # get start cell id and end cell id
    ret = ret.join(cells_existed.set_index('hex_id'), on='start_cell_hex').rename(columns={'cell_id': 'start_cell'})
    ret = ret.join(cells_existed.set_index('hex_id'), on='end_cell_hex').rename(columns={'cell_id': 'end_cell'})
    ret = ret.rename(columns={'starttime': 'start_time', 'start station id': 'start_station', 'end station id': 'end_station', 'tripduration': 'duration'})
    ret = ret[['start_time', 'start_station', 'end_station', 'duration', 'gender', 'usertype', 'start_cell', 'end_cell']]

    #   get in data cell hex_id
    data_mapping_data = cells_existed[['cell_id', 'hex_id']]
    #   get neighbors in cell bool matrix
    data_neighbor = pd.DataFrame(0, index=data_mapping_data['cell_id'], columns=np.arange(6), dtype=np.int64)
    #   get cells without neighbors
    drop_mapping_data = data_neighbor[data_neighbor.sum(axis=1) == -6]
    drop_mapping_data["cell_id"] = pd.to_numeric(drop_mapping_data.index)
    print(drop_mapping_data)
    before = len(ret)
    # drop cells have no neighbors
    ret.drop(ret[ret['start_cell'].isin(drop_mapping_data['cell_id']) | ret['end_cell'].isin(drop_mapping_data['cell_id'])].index, axis=0, inplace=True)
    after = len(ret)
    print(f"{before - after}/{before} rows droped {((before - after)/before * 100):.2f}%")
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

    return cells_existed


if __name__ == "__main__":
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    station_file_path = sys.argv[3]
    cells_file_path = 'none'
    if len(sys.argv) > 4:
        cells_file_path = sys.argv[4]

    output_data_path = os.path.join(output_folder, data_file_name)
    cell_init_file_path = os.path.join(output_folder, cell_init_file_name)
    neighbor_file_path = os.path.join(output_folder, neighbor_file_name)
    cell_to_hex_file_path = os.path.join(output_folder, cell_to_hex_file_name)

    # read existed cells : cell_id , hex_id
    cells_existed = _read_exist_cells(cells_file_path)
    full_cells_init = _read_cell_init(station_file_path)

    data_cell_dfs = []
    # generate bin file
    for src_file in input_file_list:
        src_full_path = os.path.join(input_folder, src_file)

        # show current process file
        print(f"processing {src_full_path}")

        r = read_src_file(src_full_path)

        if r is not None and len(r) > 0:
            cells_existed = concat(r, output_data_path, cells_existed)

    # filter cell by data
    data_cell_init = cells_existed.join(full_cells_init[['station_id', 'capacity', 'init']].set_index('station_id'), on='hex_id')

    data_cell_name = cells_existed[['cell_id', 'hex_id']]

    data_mapping_data = cells_existed[['cell_id', 'hex_id']]

    data_cell_init = data_cell_init[['cell_id', 'capacity', 'init']]

    # generate cell neighbors
    data_neighbor = pd.DataFrame(0, index=data_mapping_data['cell_id'], columns=np.arange(6), dtype=np.int64)

    # write cell init file
    with open(cell_init_file_path, mode="w", encoding="utf-8", newline='') as cell_init_file:
        data_cell_init.to_csv(cell_init_file, index=False)

    # write cell neighbors file
    with open(neighbor_file_path, mode="w", encoding="utf-8", newline='') as mapping_file:
        data_neighbor.to_csv(mapping_file, index=False, header=False)

    # write cell name file
    with open(cell_to_hex_file_path, mode="w", encoding="utf-8", newline='') as cell_to_hex_file:
        data_cell_name.to_csv(cell_to_hex_file, index=False)
