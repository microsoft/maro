# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import os
import threading
from datetime import datetime
from math import floor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


class DumpConverter:
    """ This class is used for convert binary snapshot dump content to CSV format. """

    def __init__(self, parent_path="", scenario_name="", prefix="epoch_", serial=0):
        super().__init__()
        self._parent_path = parent_path
        self._prefix = prefix
        self._serial = serial
        self._scenario_name = scenario_name
        self._manifest_created = False
        self._mapping_created = False

    def _generate_new_folder(self, parent_path):
        now = datetime.now()
        self._foldername = "snapshot_dump_" + now.strftime("%Y_%m_%d_%H_%M_%S_%f")[:-3]
        if parent_path != "":
            dir_path = Path(os.path.dirname(parent_path))
            if not dir_path.exists():
                return
            if not Path(parent_path).exists():
                os.mkdir(parent_path)
            self._foldername = os.path.join(parent_path, self._foldername)

        folder_path = Path(self._foldername)
        if folder_path.exists():
            return
        os.mkdir(self._foldername)

    @property
    def dump_folder(self):
        return self._foldername

    @property
    def current_serial(self):
        return self._serial

    @property
    def scenario_name(self):
        return self._scenario_name

    def reset_folder_path(self):
        self._generate_new_folder(self._parent_path)
        self._serial = 0

    def get_new_snapshot_folder(self):
        folder = os.path.join(self._foldername, self._prefix + str(self._serial))
        os.mkdir(folder)
        self._serial = self._serial + 1
        self._last_snapshot_folder = folder
        return folder

    def process_data(self, config_data: dict):
        for cur_dir, dirs, files in os.walk(self._last_snapshot_folder):
            for file in files:
                if file.endswith(".meta"):
                    col_info_dict = self.get_column_info(os.path.join(cur_dir, file))
                    data = np.load(os.path.join(cur_dir, file.replace(".meta", ".npy")))

                    frame_idx = 0
                    csv_data = []
                    for frame in data:
                        node_idx = 0
                        file_name = file.replace(".meta", "")
                        for node in frame:
                            node_dict = {"frame_index": frame_idx, "name": file_name + "_" + str(node_idx)}
                            col_idx = 0
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
                    dataframe.to_csv(os.path.join(cur_dir, file.replace(".meta", ".csv")), index=False)

        self.save_manifest_file(config_data)

    def start_processing(self, config_data: dict):
        thread = threading.Thread(target=self.process_data, args=(config_data,))
        thread.start()

    def get_column_info(self, filename):
        with open(filename, "r") as f:
            columns = f.readline().strip()
            elements = f.readline().strip()
            f.close()

        col_dict = {}
        cols = str.split(columns, ",")
        element_list = str.split(elements, ",")
        i = 0
        for col in cols:
            col_dict[col] = element_list[i]
            i = i + 1
        return col_dict

    def clear_raw_data(self):
        for _, dirs, files in os.walk(self._foldername):
            for file in files:
                if file.endswith(".meta") or file.endswith(".npy"):
                    os.remove(file)

    def dump_descsion_events(self, decision_events, start_tick: int, resolution: int):
        if 0 == len(decision_events):
            return
        decision_events_file = os.path.join(self._last_snapshot_folder, "decision_events.csv")
        headers, columns_count = self._calc_event_headers(decision_events[0])
        array = []
        for event in decision_events:
            key = event.__getstate__()
            if key.__contains__("tick"):
                frame_idx = floor((key["tick"] - start_tick) / resolution)
                key["frame_idx"] = frame_idx
            array.append(key)

        dataframe = pd.DataFrame(array)
        frameidx = dataframe.frame_idx
        dataframe = dataframe.drop("frame_idx", axis=1)
        dataframe.insert(0, "frame_idx", frameidx)
        dataframe.to_csv(decision_events_file, index=False)

    def _calc_event_headers(self, event):
        if event is None:
            return [], 0
        headers = []
        count = 0
        for attr in dir(event):
            if attr[0] != "_":
                headers.append(attr)
                count = count + 1

        return headers, count

    def save_manifest_file(self, config_data: dict):
        if self._scenario_name == "":
            return
        outputfile = os.path.join(self._foldername, "manifest.yml")
        if os.path.exists(outputfile):
            manifest_content = {}
            with open(outputfile, "r", encoding="utf-8") as manifest_file:
                manifest_content = yaml.load(manifest_file, Loader=yaml.FullLoader)
                manifest_file.close()

            manifest_content["dump_details"]["epoch_num"] = self._serial
            with open(outputfile, "w", encoding="utf-8") as new_manifest_file:
                yaml.dump(manifest_content, new_manifest_file)
                new_manifest_file.close()
            return

        content = {}
        content["scenario"] = self._scenario_name
        # mapping file.
        if config_data is not None:
            file_name = os.path.join(self._foldername, "config.yml")
            with open(file_name, "w+") as config_file:
                yaml.dump(config_data, config_file)
                config_file.close()
            content["mappings"] = os.path.basename(file_name)

        dump_details = {}
        meta_file_list = []
        for curDir, dirs, files in os.walk(self._last_snapshot_folder):
            for file in files:
                if file.endswith(".meta"):
                    meta_file_list.append(file.replace(".meta", ".csv"))

        dump_details["prefix"] = self._prefix
        dump_details["metafiles"] = meta_file_list
        dump_details[self._prefix + "num"] = self._serial
        content["dump_details"] = dump_details
        with open(outputfile, "w", encoding="utf-8") as f:
            yaml.dump(content, f)
            f.close()
        self._manifest_created = True
