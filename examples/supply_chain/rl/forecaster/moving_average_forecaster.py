# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pandas as pd

from .base_forecaster import BaseForecaster


class MovingAverageForecaster(BaseForecaster):
    def forecast_future_demand(self, state: dict, df_history: pd.DataFrame) -> pd.DataFrame:
        history_demand_mean = df_history["Demand"].mean().item()
        future_demand = [history_demand_mean] * self.data_loader_conf["future_len"]
        return future_demand
