import pandas as pd
import numpy as np
class BaseDataLoader(object):
    def __init__(self, data_loader_conf) -> None:
        super().__init__()
        self.data_loader_conf = data_loader_conf

    def load(self, state) -> None:
        pass

class DataLoaderFromFile(BaseDataLoader):
    def __init__(self, data_loader_conf) -> None:
        super().__init__(data_loader_conf)

        oracle_file = self.data_loader_conf["oracle_file"]
        if oracle_file.endswith(".csv"):
            self.df_raws = pd.read_csv(oracle_file)
        elif oracle_file.endswith(".tsv"):
            self.df_raws = pd.read_csv(oracle_file, sep="\t")
        elif oracle_file.endswith(".xlsx"):
            self.df_raws = pd.read_excel(oracle_file)
        else:
            raise NotImplementedError 
        
    def load(self, state):
        entity_id = state["entity_id"]
        target_df = self.df_raws[self.df_raws["entity_id"]==entity_id]
        return target_df, 0

class DataLoaderFromHistory(BaseDataLoader):
    def load(self, state):
        target_df = pd.DataFrame(columns=["price", "storage_cost", "order_cost", "demand"])

        # Including historcy and today
        history_len = min(self.data_loader_conf["history_len"] + 1, len(state["history_demand"]))
        history_range = range(len(state["history_demand"]) - history_len, len(state["history_demand"]))
        for index in history_range:
            target_df = target_df.append(pd.Series({
                'price': state["history_price"][index],
                'storage_cost':state["unit_storage_cost"], 
                'order_cost':state["unit_order_cost"], 
                'demand': state["history_demand"][index]
            }), ignore_index=True)
        
        # Use history mean represents the future
        his_mean_price = target_df["price"].mean().item()
        his_demand_price = target_df["demand"].mean().item()
        future_len = self.data_loader_conf["future_len"]
        for index in range(0, future_len):
            target_df = target_df.append(pd.Series({
                'price': his_mean_price,
                'storage_cost':state["unit_storage_cost"], 
                'order_cost':state["unit_order_cost"], 
                'demand': his_demand_price
            }), ignore_index=True)
        today_index = history_len - 1
        return target_df, today_index