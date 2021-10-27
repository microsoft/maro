# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import pickle

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


def transform_data(scaler_x: MinMaxScaler(), scaler_y: MinMaxScaler(), x, y):
    # Rescale inputs/outputs of the Neural Nets based on pre-defined scalar
    x_transformed = scaler_x.transform(x)
    y_transformed = scaler_y.transform(y)
    return x_transformed, y_transformed

def generate_dataset(
    dataset_path: str, input_filename: str, input_dim: int,
    seed: int=123, split_random: bool=True,
    train_fraction: float=0.6, validation_fraction: float=0.2,
    lower_rescale_bound: float=-1, upper_rescale_bound: float=1,
):
    # Read the CSV input CSV file under the dataset dir
    filepath = os.path.join(dataset_path, input_filename)
    df = pd.read_csv(filepath, sep=',', delimiter=None, header='infer')
    df = df.dropna()

    # Split the data records into training, validation, testing subsets.
    if split_random:
        df = df.sample(frac=1, random_state=seed)
    train, val, test = np.split(
        ary=df,
        indices_or_sections=[
            int(len(df) * train_fraction),
            int(len(df) * (train_fraction + validation_fraction))
        ]
    )

    # Split data into input x and target y
    x_train = train.iloc[:, :input_dim]
    y_train = train.iloc[:, input_dim:]
    x_val = val.iloc[:, :input_dim]
    y_val = val.iloc[:, input_dim:]
    x_test = test.iloc[:, :input_dim]
    y_test = test.iloc[:, input_dim:]

    # Using Min Max scaler to transform data
    scaler_x = MinMaxScaler(feature_range=(lower_rescale_bound, upper_rescale_bound)).fit(x_train)
    scaler_y = MinMaxScaler(feature_range=(lower_rescale_bound, upper_rescale_bound)).fit(y_train)

    x_train, y_train = transform_data(scaler_x, scaler_y, x_train, y_train)
    x_val, y_val = transform_data(scaler_x, scaler_y, x_val, y_val)
    x_test, y_test = transform_data(scaler_x, scaler_y, x_test, y_test)

    # Dump to files
    for subname in ["train", "val", "test", "scaler"]:
        os.makedirs(os.path.join(dataset_path, subname), exist_ok=True)

    for x, y, subname in zip([x_train, x_val, x_test], [y_train, y_val, y_test], ["train", "val", "test"]):
        with open(os.path.join(dataset_path, subname, f"{subname}_data.pickle"), "wb") as fp:
            pickle.dump(x, fp)
        with open(os.path.join(dataset_path, subname, f"{subname}_target.pickle"), "wb") as fp:
            pickle.dump(y, fp)

    joblib.dump(scaler_x, os.path.join(dataset_path, "scaler", "x_scaler.joblib"))
    joblib.dump(scaler_y, os.path.join(dataset_path, "scaler", "y_scaler.joblib"))

    return x_train, y_train, x_val, y_val

class AhuDataset(Dataset):
    def __init__(self, x, y):
        self._x = x
        self._y = y

    def __getitem__(self, index):
        return self._x[index].astype(np.float32), self._y[index].astype(np.float32)

    def __len__(self):
        return len(self._x)
