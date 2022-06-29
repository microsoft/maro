# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pandas as pd

class BaseForecastor(object):
    def __init__(self, data_loader_conf) -> None:
        self.data_loader_conf = data_loader_conf

    def forecast_future_demand(self, state: dict, history_df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError