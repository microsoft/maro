# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod
import pandas as pd


class BaseForecaster(object):
    def __init__(self, data_loader_conf: dict) -> None:
        self.data_loader_conf = data_loader_conf

    @abstractmethod
    def forecast_future_demand(self, state: dict, df_history: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError
