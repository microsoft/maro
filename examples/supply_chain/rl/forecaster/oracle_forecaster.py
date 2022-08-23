# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import datetime
import os

import pandas as pd

from .base_forecaster import BaseForecaster


class OracleForecaster(BaseForecaster):
    """
    The oracle forecaster load data from form spreadsheets. Each form needs to include the following columns.
    FacilityName: facility name, should be defined in config.yml
    SkuName: sku name, should be defined in config.yml
    Date: date in format YYYY-MM-DD
    Price: sku's mean price to end-customers in this store
    Demand: sku's demand to end-customers in this store.
    Cost: sku's mean cost for upstream facility in this store.
    """

    def __init__(self, data_loader_conf: dict) -> None:
        super().__init__(data_loader_conf)

        oracle_file_dir = self.data_loader_conf["oracle_file_dir"]
        df_list = []
        for oracle_name in os.listdir(oracle_file_dir):
            # TODO: The simulator will produce the preprocessed file in the target dir.
            # To avoid the conflict, currently the file name with preprocessed will be skipped.
            if "preprocessed" not in oracle_name:
                oracle_file_path = os.path.join(oracle_file_dir, oracle_name)
                if oracle_file_path.endswith(".csv"):
                    df_list.append(pd.read_csv(oracle_file_path, parse_dates=["Date"]))
                elif oracle_file_path.endswith(".tsv"):
                    df_list.append(pd.read_csv(oracle_file_path, parse_dates=["Date"], sep="\t"))
                elif oracle_file_path.endswith(".xlsx"):
                    df_list.append(pd.read_excel(oracle_file_path, parse_dates=["Date"]))
                else:
                    # TODO: Whether to terminate the program or just skip the file need to be decided.
                    raise NotImplementedError
        self.df_raws = pd.concat(df_list, axis=0)
        self.df_raws.sort_values(by="Date", ascending=True)

    def forecast_future_demand(self, state: dict, df_history: pd.DataFrame) -> pd.DataFrame:
        """
        The OracleForecaster filters data from the read-in file by the info in the given state. The required state keys include:
        sku_name: target sku name for forecasting.
        facility_name: target facility name for forecasting
        tick: date tick to indicate the days
        start_date_time: start date for simulation
        durations: time span in days
        """
        sku_name = state["sku_name"]
        facility_name = state["facility_name"]
        future_start_offset = state["tick"] + 1
        future_start_date = state["start_date_time"] + datetime.timedelta(future_start_offset)
        future_end_offset = min(state["tick"] + self.data_loader_conf["future_len"], state["durations"])
        future_end_date = state["start_date_time"] + datetime.timedelta(future_end_offset)
        df_target = self.df_raws[
            (self.df_raws["FacilityName"] == facility_name)
            & (self.df_raws["SkuName"] == sku_name)
            & (self.df_raws["Date"] >= future_start_date)
            & (self.df_raws["Date"] <= future_end_date)
        ]
        future_demand = df_target.sort_values(by=["Date"])["Demand"].values
        return future_demand
