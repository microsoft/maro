# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pandas as pd
import datetime


class BaseDataLoader(object):
    def __init__(self, data_loader_conf) -> None:
        super().__init__()
        self.data_loader_conf = data_loader_conf

    def load(self, state: dict) -> None:
        pass


class DataLoaderFromFile(BaseDataLoader):
    def __init__(self, data_loader_conf) -> None:
        super().__init__(data_loader_conf)

        oracle_file = self.data_loader_conf["oracle_file"]
        if oracle_file.endswith(".csv"):
            self.df_raws = pd.read_csv(oracle_file, parse_dates=["Date"])
        elif oracle_file.endswith(".tsv"):
            self.df_raws = pd.read_csv(oracle_file, parse_dates=["Date"], sep="\t")
        elif oracle_file.endswith(".xlsx"):
            self.df_raws = pd.read_excel(oracle_file, parse_dates=["Date"])
        else:
            raise NotImplementedError
        self.df_raws.sort_values(by="Date", ascending=True)
        self.start_date = min(self.df_raws["Date"])
        self.end_date = max(self.df_raws["Date"])
        self.date_len = len(pd.date_range(self.start_date, self.end_date))

    def load(self, state: dict) -> pd.DataFrame:
        SKU_id = state["SKU"]
        history_offset = max(state["tick"] - self.data_loader_conf["history_len"], 0)
        history_start_date = self.start_date + datetime.timedelta(history_offset)
        future_offset = min(state["tick"] + self.data_loader_conf["future_len"], self.date_len)
        future_end_date = self.start_date + datetime.timedelta(future_offset)
        target_df = self.df_raws[
            (self.df_raws["SKU"] == SKU_id)
            & (self.df_raws["Date"] >= history_start_date)
            & (self.df_raws["Date"] <= future_end_date)
        ]
        return target_df.sort_values(by=["Date"])


class DataLoaderFromHistory(BaseDataLoader):
    def load(self, state: dict) -> pd.DataFrame:
        target_df = pd.DataFrame(columns=["Price", "Cost", "Demand"])

        # Including history and today
        history_start = max(state["tick"] - self.data_loader_conf["history_len"], 0)
        for index in range(history_start, state["tick"] + 1):
            target_df = target_df.append(pd.Series({
                "Price": state["history_price"][index],
                "Cost": state["unit_order_cost"],
                "Demand": state["history_demand"][index]
            }), ignore_index=True)

        # Use history mean represents the future
        his_mean_price = target_df["Price"].mean().item()
        his_demand_price = target_df["Demand"].mean().item()
        future_len = self.data_loader_conf["future_len"]
        for index in range(0, future_len):
            target_df = target_df.append(pd.Series({
                "Price": his_mean_price,
                "Cost": state["unit_order_cost"],
                "Demand": his_demand_price
            }), ignore_index=True)
        return target_df
