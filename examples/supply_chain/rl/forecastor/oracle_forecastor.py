# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import datetime
import os

import pandas as pd

from .base_forecastor import BaseForecastor


class OracleForecastor(BaseForecastor):
    def __init__(self, data_loader_conf) -> None:
        super().__init__(data_loader_conf)

        oracle_file_dir = self.data_loader_conf["oracle_file_dir"]
        df_list = []
        for oracle_name in os.listdir(oracle_file_dir):
            # The simulator will product the preprocessed file in the target dir.
            # To avoid the confilct, currently the file name with preprocessed will be skipped.
            if "preprocessed" not in oracle_name:
                oracle_file_path = os.path.join(oracle_file_dir, oracle_name)
                if oracle_file_path.endswith(".csv"):
                    df_list.append(pd.read_csv(oracle_file_path, parse_dates=["Date"]))
                elif oracle_file_path.endswith(".tsv"):
                    df_list.append(pd.read_csv(oracle_file_path, parse_dates=["Date"], sep="\t"))
                elif oracle_file_path.endswith(".xlsx"):
                    df_list.append(pd.read_excel(oracle_file_path, parse_dates=["Date"]))
                else:
                    raise NotImplementedError
        self.df_raws = pd.concat(df_list, axis=0)
        self.df_raws.sort_values(by="Date", ascending=True)
        self.start_date = min(self.df_raws["Date"])
        self.end_date = max(self.df_raws["Date"])
        self.date_len = len(pd.date_range(self.start_date, self.end_date))

    def forecast_future_demand(self, state: dict, history_df: pd.DataFrame) -> pd.DataFrame:
        sku_name = state["sku_name"]
        facility_name = state["facility_name"]
        future_start_offset = state["tick"] + 1
        future_start_date = self.start_date + datetime.timedelta(future_start_offset)
        future_end_offset = min(state["tick"] + self.data_loader_conf["future_len"], self.date_len)
        future_end_date = self.start_date + datetime.timedelta(future_end_offset)
        target_df = self.df_raws[
            (self.df_raws["FacilityName"] == facility_name)
            & (self.df_raws["SkuName"] == sku_name)
            & (self.df_raws["Date"] >= future_start_date)
            & (self.df_raws["Date"] <= future_end_date)
        ]
        future_demand = target_df.sort_values(by=["Date"])["Demand"].values
        return future_demand
