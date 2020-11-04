import os
import pandas as pd
import csv
import numpy as np
import tqdm
import json
import yaml
# delete a folder recursively
def delete_dir(root_path):
    ls = os.listdir(root_path)
    for i in ls:
        c_path = os.path.join(root_path, i)
        if os.path.isdir(c_path):
            delete_dir(c_path)
            os.rmdir(c_path)
        else:
            os.remove(c_path)


# clean & init summary csv file
def init_csv(file_path, header):
    if os.path.exists(file_path):
        os.remove(file_path)
    with open(file_path, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()


# calculate summary info and generate corresponding csv file
def summary_append(dir_epoch, file_name, header, sum_dataframe, i, output_path):
    data = pd.read_csv(os.path.join(dir_epoch, file_name))
    data_insert = []
    for ele in header:
        data_insert.append(np.sum(np.array(data[ele]), axis=0))
    sum_dataframe.loc[i] = data_insert
    sum_dataframe.to_csv(output_path, header=True, index=True)

# generate summary info of a senario
def generate_summary(senario, ROOT_PATH, ports_file_path, vessels_file_path, stations_file_path):
    ports_header = ['capacity', 'empty', 'full', 'on_shipper', 'on_consignee', 'shortage', 'booking', 'fulfillment']
    vessels_header = ['capacity', 'empty', 'full', 'remaining_space', 'early_discharge']
    stations_header = ['bikes', 'shortage', 'trip_requirement', 'fulfillment', 'capacity']
    dbtype_list_all = os.listdir(ROOT_PATH)

    temp_len = len(dbtype_list_all)
    dbtype_list = []
    for index in range(0, temp_len):
        if (os.path.exists(os.path.join(ROOT_PATH, r'snapshot_' + str(index)))):
            dbtype_list.append(os.path.join(ROOT_PATH, r'snapshot_' + str(index)))

    if senario == 'CIM':
        init_csv(ports_file_path, ports_header)
        #init_csv(vessels_file_path, vessels_header)
        ports_sum_dataframe = pd.read_csv(ports_file_path, names=ports_header)

        # vessels_sum_dataframe = pd.read_csv(vessels_file_path, names=vessels_header)
    else:
        init_csv(stations_file_path, stations_header)
        stations_sum_dataframe = pd.read_csv(stations_file_path, names=stations_header)
    if senario=='CIM':
        i = 1
        for i in tqdm.tqdm(range(len(dbtype_list))):
            dbtype=dbtype_list[i]
            dir_epoch = os.path.join(ROOT_PATH, dbtype)
            if not os.path.isdir(dir_epoch):
                continue
            if senario == 'CIM':
                summary_append(dir_epoch, 'ports.csv', ports_header, ports_sum_dataframe, i, ports_file_path)
                # summary_append(dir_epoch, 'vessels.csv', vessels_header, vessels_sum_dataframe, i,vessels_file_path)
                i = i + 1
    elif senario=='CITI_BIKE':
        data=pd.read_csv(os.path.join(ROOT_PATH,'snapshot_0','stations.csv'))
        data=data[['bikes', 'trip_requirement','fulfillment','capacity']].groupby(data['name']).sum()
        data['fulfillment_ratio']=list(map(lambda x,y: float('{:.4f}'.format(x/(y+1/1000))), data['fulfillment'], data['trip_requirement']))
        data.to_csv(stations_file_path)


# rename snapshot folders in the mode of "snapshot_1","snapshot_2" etc
def rename_data(ROOT_PATH):
    dbtype_list = os.listdir(ROOT_PATH)
    dbtype_list.sort()
    b = 0
    for i in tqdm.tqdm(range(len(dbtype_list))):
        name=dbtype_list[i]
        if not os.path.isdir(os.path.join(ROOT_PATH, name)):
            continue
        if name[0:8] != 'snapshot':
            continue
        newname = 'snapshot_' + str(b)
        b = b + 1
        if not os.path.exists(os.path.join(ROOT_PATH, newname)):
            os.rename(os.path.join(ROOT_PATH, name), os.path.join(ROOT_PATH, newname))


# generate down pooling list based on origin data and down pooling rate
def generate_down_pooling_sample(origin_len, down_pooling_len, start_epoch, end_epoch):
    down_pooling_range = range(start_epoch, end_epoch, down_pooling_len)
    down_pooling_range = list(down_pooling_range)
    if end_epoch not in down_pooling_range:
        down_pooling_range.append(end_epoch)
    return down_pooling_range

def get_holder_name_conversion(ROOT_PATH,CONVER_PATH):
    if os.path.exists(os.path.join(ROOT_PATH,r'name_conversion.csv')):
        os.remove(os.path.join(ROOT_PATH,r'name_conversion.csv'))
    filename,type = os.path.splitext(CONVER_PATH)
    if type=='.json':
        with open(CONVER_PATH, 'r', encoding='utf8')as fp:
            json_data = json.load(fp)
            name_list=[]
            for item in json_data['data']['stations']:
                name_list.append(item['name'])
            df=pd.DataFrame(name_list)
            df.to_csv(os.path.join(ROOT_PATH,r'name_conversion.csv'),index=False)
    else:
        f = open(CONVER_PATH, 'r')
        ystr = f.read()
        aa = yaml.load(ystr, Loader=yaml.FullLoader)
        key_list=aa['ports'].keys()
        df = pd.DataFrame(list(key_list))
        df.to_csv(os.path.join(ROOT_PATH, r'name_conversion.csv'), index=False)

def start_vis(input: str, conver_path:str,**kwargs):
    try:
        import streamlit as st
    except ImportError:
        os.system('pip install streamlit')
    try:
        import tqdm
    except ImportError:
        os.system('pip install tqdm')
    ROOT_PATH = input
    CONVER_PATH=conver_path
    if not os.path.exists(ROOT_PATH):
        print ('path not exist')
        os._exit(0)
    # path to restore summary files
    ports_file_path = os.path.join(ROOT_PATH, r'snapshot_ports_summary.csv')
    vessels_file_path = os.path.join(ROOT_PATH, r'snapshot_vessels_summary.csv')
    stations_file_path = os.path.join(ROOT_PATH, r'snapshot_stations_summary.csv')
    if os.path.exists(ports_file_path) or os.path.exists(stations_file_path):
        print ('Data is generated. Display charts directly.')

    print("rename data")
    rename_data(ROOT_PATH)
    print("rename data done")
    if os.path.exists(os.path.join(ROOT_PATH, 'snapshot_0', 'ports.csv')):
        senario = 'CIM'
    else:
        senario = 'CITI_BIKE'
    print("generate summary")
    generate_summary(senario, ROOT_PATH, ports_file_path, vessels_file_path, stations_file_path)
    print("generate summary done")
    if CONVER_PATH is not None:
        print("generate name conversion")
        get_holder_name_conversion(ROOT_PATH,CONVER_PATH)
        print("generate name conversion done")
    os.system('streamlit cache clear')
    os.system(r'streamlit run ' + os.path.join(
        os.getcwd() + r'\maro\cli\inspector\visualization.py ') + r'-- ' +ROOT_PATH)



