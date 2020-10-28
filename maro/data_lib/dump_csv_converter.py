# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import os
from datetime import datetime
from pathlib import Path
import threading
import pandas as pd
import numpy as np
import inspect


class dump_csv_converter:
    """ This class is used for convert binary snapshot dump content to CSV format. """

    def __init__(self, parent_path = ''):
        super().__init__()
        self.__parent_path = parent_path
        #self.__generate_new_folder(parent_path)

    def __generate_new_folder(self, parent_path):
        now = datetime.now()
        self._foldername = 'snapshot_dump_' + now.strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
        if parent_path is not '':
            self._foldername = os.path.join(parent_path, self._foldername)

        folderPath = Path(self._foldername)
        if folderPath.exists():
            return
        os.mkdir(self._foldername)

    @property
    def dump_folder(self):
        return self._foldername

    def reset_folder_path(self):
        self.__generate_new_folder(self.__parent_path)

    def process_data(self):
        for curDir, dirs, files in os.walk(self._foldername):
            for file in files:
                if file.endswith('.meta'):
                    col_info_dict = self.get_column_info(os.path.join(curDir, file))
                    data = np.load(os.path.join(curDir, file.replace('.meta', '.npy')))

                    frame_idx = 0
                    csv_data = []
                    for frame in data:
                        node_idx = 0
                        for node in frame:
                            node_dict = {'frame_index': frame_idx, 'name': file.replace('.meta', '') + '_' + str(node_idx)}
                            col_idx = 0;
                            for key in col_info_dict.keys():
                                if col_info_dict[key] == 1:
                                    node_dict[key] = node[col_idx]
                                else:
                                    node_dict[key] = str(node[col_idx])
                                col_idx = col_idx + 1

                            node_idx = node_idx + 1
                            csv_data.append(node_dict)

                        frame_idx = frame_idx + 1

                    dataframe = pd.DataFrame(csv_data)
                    dataframe.to_csv(os.path.join(curDir, file.replace('.meta', '.csv')), index = False)

    def start_processing(self):
        thread = threading.Thread(target = self.process_data)
        thread.start()

    def get_column_info(self, filename):
        with open(filename, 'r') as f:
            columns = f.readline().replace('\n', '').replace('\r\n', '')
            elements = f.readline().replace('\n', '').replace('\r\n', '')

        col_dict = {}
        cols = str.split(columns, ',')
        eles = str.split(elements, ',')
        i = 0
        for col in cols:
            col_dict[col] = eles[i]
            i = i + 1
        return col_dict

    def clear_raw_data(self):
        for curDir, dirs, files in os.walk(self._foldername):
            for file in files:
                if file.endswith('.meta') or file.endswith('.npy'):
                    os.remove(file)

    def dump_descsion_events(self, decision_events):
        decision_events_file = os.path.join(self._foldername, 'decision_events.csv')
        for e in decision_events:
            print(e)
        headers, colums_count = self._calc_event_headers(decision_events[0])
        array = []
        for event in decision_events:
            attr_dict = dict()
            for key in headers:
                if event.__dict__.__contains__(key):
                    if key == 'snapshot_list':
                        obj = event.__dict__[key]
                        for obj_key in obj.__dict__.keys:
                            if obj_key[0] is not '_':
                                attr_dict[obj_key] = obj.__dict__[obj_key]
                    else:
                        attr_dict[key] = event.__dict__[key]

            array.append(attr_dict)

        dataframe = pd.DataFrame(array)
        dataframe.to_csv(decision_events_file, index = False)

    def _calc_event_headers(self, event):
        if event is None:
            return [], 0
        headers = []
        count = 0
        for attr in dir(event):
            if attr[0] is not '_':
                headers.append(attr)
                count = count + 1

        return headers, count


