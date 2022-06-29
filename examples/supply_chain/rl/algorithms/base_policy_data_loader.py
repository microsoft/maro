# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import datetime
import pandas as pd


class BaseDataLoader(object):
    def __init__(self, data_loader_conf) -> None:
        super().__init__()
        self.data_loader_conf = data_loader_conf

    def load(self, state: dict) -> None:
        pass


class OracleDataLoader(BaseDataLoader):
    def __init__(self, data_loader_conf) -> None:
        super().__init__(data_loader_conf)

        oracle_file_dir = self.data_loader_conf["oracle_file_dir"]
        oracle_files = self.data_loader_conf["oracle_files"]
        df_list = []
        for oracle_name in oracle_files:
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
        # self.start_date = min(self.df_raws["Date"])
        self.start_date = datetime.datetime.strptime(self.data_loader_conf["start_date_time"], '%Y-%m-%d')
        self.end_date = max(self.df_raws["Date"])
        self.durations = self.data_loader_conf["durations"]

    def load(self, state: dict) -> pd.DataFrame:
        sku_name = state["sku_name"]
        facility_name = state["facility_name"]
        history_offset = max(state["tick"] - self.data_loader_conf["history_len"], 0)
        history_start_date = self.start_date + datetime.timedelta(history_offset)
        future_offset = min(state["tick"] + self.data_loader_conf["future_len"], self.durations)
        future_end_date = self.start_date + datetime.timedelta(future_offset)
        target_df = self.df_raws[
            (self.df_raws["FacilityName"] == facility_name)
            & (self.df_raws["SkuName"] == sku_name)
            & (self.df_raws["Date"] >= history_start_date)
            & (self.df_raws["Date"] <= future_end_date)
        ]
        return target_df.sort_values(by=["Date"])


class MovingAverageDataLoader(BaseDataLoader):
    def load(self, state: dict) -> pd.DataFrame:
        cost = state["upstream_price_mean"]
        target_df = pd.DataFrame(columns=["Price", "Cost", "Demand"])

        # Including history and today
        history_start = max(state["tick"] - self.data_loader_conf["history_len"], 0)
        for index in range(history_start, state["tick"] + 1):
            target_df = target_df.append(pd.Series({
                "Price": state["history_price"][index],
                "Cost": cost, #state["unit_order_cost"],
                "Demand": state["history_demand"][index]
            }), ignore_index=True)

        # Use history mean represents the future
        his_price_mean = target_df["Price"].mean().item()
        his_demand_mean = target_df["Demand"].mean().item()
        future_len = self.data_loader_conf["future_len"]
        for index in range(0, future_len):
            target_df = target_df.append(pd.Series({
                "Price": his_price_mean,
                "Cost": cost,
                "Demand": his_demand_mean
            }), ignore_index=True)
        return target_df


class ForecastingDataLoader(BaseDataLoader):
    def __init__(self, data_loader_conf) -> None:
        super().__init__(data_loader_conf)
        forecasting_file_dir = self.data_loader_conf["forecasting_file_dir"]
        forecasting_files = [i for i in os.listdir(forecasting_file_dir) if "preprocessed" not in i]
        df_list = []
        for forecasting_name in forecasting_files:
            forecasting_file_path = os.path.join(forecasting_file_dir, forecasting_name)
            if forecasting_file_path.endswith(".csv"):
                df_list.append(pd.read_csv(forecasting_file_path, parse_dates=["Date"]))
            elif forecasting_file_path.endswith(".tsv"):
                df_list.append(pd.read_csv(forecasting_file_path, parse_dates=["Date"], sep="\t"))
            elif forecasting_file_path.endswith(".xlsx"):
                df_list.append(pd.read_excel(forecasting_file_path, parse_dates=["Date"]))
            else:
                raise NotImplementedError
        self.df_raws = pd.concat(df_list, axis=0)
        self.df_raws.sort_values(by="Date", ascending=True)
        self.start_date = datetime.datetime.strptime(self.data_loader_conf["start_date_time"], '%Y-%m-%d')
        self.durations = len(pd.date_range(self.start_date, self.end_date))
    
    def load(self, state: dict) -> pd.DataFrame:
        cost = state["upstream_price_mean"]
        # Including history and today from env
        history_start = max(state["tick"] - self.data_loader_conf["history_len"], 0)
        target_df_his = pd.DataFrame(columns=["Price", "Cost", "Demand"])
        for index in range(history_start, state["tick"] + 1):
            target_df_his = target_df_his.append(pd.Series({
                "Price": state["history_price"][index],
                "Cost": cost, 
                "Demand": state["history_demand"][index]
            }), ignore_index=True)
        
        # Load predict from forecasting result file.
        current_date = self.start_date + datetime.timedelta(state["tick"])
        if self.data_loader_conf["future_len"] == 0:
            future_offset = 8
        else:
            future_offset = min(state["tick"] + self.data_loader_conf["future_len"], self.durations)
        future_end_date = self.start_date + datetime.timedelta(future_offset)
        sku_name = state["sku_name"]
        facility_name = state["facility_name"]
        target_df_fut = self.df_raws[
            (self.df_raws["FacilityName"] == facility_name)
            & (self.df_raws["SkuName"] == sku_name)
            & (self.df_raws["Date"] > current_date)
            & (self.df_raws["Date"] <= future_end_date)
        ]
        target_df_fut["Demand"][target_df_fut["Date"] <= datetime.datetime(2021, 7, 7)] = target_df_fut["Sold"][target_df_fut["Date"] <= datetime.datetime(2021, 7, 7)] 
        target_df_fut = target_df_fut[["Price", "Cost", "Demand"]]
        target_df = pd.concat([target_df_his, target_df_fut], axis=-0)
        return target_df