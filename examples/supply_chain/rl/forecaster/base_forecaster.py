# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pandas as pd


class BaseForecaster(object):
    def __init__(self, data_loader_conf: dict) -> None:
        self.data_loader_conf = data_loader_conf

    def forecast_future_demand(self, state: dict, df_history: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError
