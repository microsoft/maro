# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pandas as pd

from .base_forecastor import BaseForecastor

class MovingAverageForecastor(BaseForecastor):
    def forecast_future_demand(self, state: dict, history_df) -> pd.DataFrame:
        his_demand_mean = history_df["Demand"].mean().item()
        future_demand = [his_demand_mean] * self.data_loader_conf["future_len"]
        return future_demand
