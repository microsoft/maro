# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

import numpy as np
import torch
from shutil import copyfile
from torch.utils.data import DataLoader

from maro.utils import LogFormat, Logger

from maro.simulator.scenarios.hvac.ahu_prediction.config import data_config, fcn_config, training_config
from maro.simulator.scenarios.hvac.ahu_prediction.data_processing import AhuDataset, generate_dataset
from maro.simulator.scenarios.hvac.ahu_prediction.model import FCN_model


def train():
    logger = Logger(tag="Train", dump_folder=training_config["checkpoint_dir"], format_=LogFormat.simple)

    logger.info(f"[Data Generation] Generating dataset with config: {data_config}")
    x_train, y_train, x_val, y_val = generate_dataset(
        input_dim=fcn_config["input_dim"],
        **data_config
    )
    train_dataloader = DataLoader(
        dataset=AhuDataset(x_train, y_train),
        batch_size=training_config["batch_size"],
        shuffle=training_config["shuffle_data"]
    )
    val_dataloader = DataLoader(
        dataset=AhuDataset(x_val, y_val),
        batch_size=training_config["batch_size"],
        shuffle=training_config["shuffle_data"]
    )

    logger.info(f"[Model] Building model with config: {fcn_config}")
    model = FCN_model(**fcn_config)
    optimizer = torch.optim.Adamax(
        params=model.parameters(),
        lr=training_config["learning_rate"],
    )

    logger.info(f"[Training] Training with config: {training_config}")
    best_ep, best_loss, min_val = -1, 100, 100
    for epoch in range(training_config["epoch"]):
        model.train()
        train_loss = []
        for x, y in train_dataloader:
            y_pred = model(x)
            loss = training_config["loss"](input=y_pred, target=y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.cpu().data)

        model.eval()
        val_metric = []
        for x, y in val_dataloader:
            y_pred = model(x)
            metric = training_config["loss"](input=y_pred, target=y)
            val_metric.append(metric.cpu().data)

        mean_loss = np.mean(train_loss)
        mean_metric = np.mean(val_metric)
        logger.info(f"epoch: {epoch:4}, train: {mean_loss:.3}, val: {mean_metric:.3}")

        if mean_metric < min_val:
            best_ep = epoch
            best_loss = mean_loss
            min_val = mean_metric

            torch.save(model.state_dict(), os.path.join(training_config["checkpoint_dir"], "best_model.pt"))

        if epoch - best_ep >= training_config["early_stopping_patience"]:
            break

    logger.info(f"Training finished, best model generated in epoch: {best_ep}")
    copyfile(
        src=os.path.join(training_config["checkpoint_dir"], "best_model.pt"),
        dst=os.path.join(
            training_config["checkpoint_dir"],
            f"best_model_ep{best_ep}_loss{best_loss:.3}_val{min_val:.3}.pt"
        )
    )


def get_test_func(model_path: str, x_scaler_path: str, y_scaler_path: str, **kwargs):
    model = FCN_model(**fcn_config)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    import joblib
    x_scaler = joblib.load(x_scaler_path)
    y_scaler = joblib.load(y_scaler_path)

    def predict(sps: float, das: float, mat: float=None, mat_das_delta: float=None):
        assert (mat is not None) or (mat_das_delta is not None)
        mat_das_delta = mat - das if mat is not None else mat_das_delta
        x = np.array([sps, das, mat_das_delta]).astype(np.float32).reshape(1, -1)
        x = torch.tensor(x_scaler.transform(x))
        y_pred = model(x).detach().numpy()
        y_pred = y_scaler.inverse_transform(y_pred)[0]
        kw, at, dat = y_pred
        return {"kw": kw, "at": at, "dat": dat}

    return predict

def test():
    # Get test func
    from maro.simulator.scenarios.hvac.ahu_prediction.config import test_config
    test_func = get_test_func(**test_config)

    # Get data
    import pandas as pd
    df = pd.read_csv(test_config["filepath"], sep=",", delimiter=None, header='infer')
    df = df.dropna()
    df = df.reset_index()

    keymap = test_config["key_map"]
    required_keys = ["sps", "das", "kw", "at", "dat"]
    assert "mat_das_delta" in keymap or "mat" in keymap
    for key in required_keys:
        assert key in keymap

    data = {key: df[keymap[key]].to_numpy() for key in required_keys}
    data["mat_das_delta"] = (
        df[keymap["mat_das_delta"]].to_numpy()
        if "mat_das_delta" in keymap
        else df[keymap["mat"]].to_numpy() - df[keymap["das"]].to_numpy()
    )
    data["mat"] = (
        df[keymap["mat"]].to_numpy()
        if "mat" in keymap
        else df[keymap["das"]].to_numpy() + df[keymap["mat_das_delta"]].to_numpy()
    )

    # Predict
    data_pred = {"kw": [], "at": [], "dat": []}
    for i in range(len(df)):
        pred = test_func(sps=data["sps"][i], das=data["das"][i], mat_das_delta=data["mat_das_delta"][i])
        for key, value in pred.items():
            data_pred[key].append(value)

    # Visualization
    import matplotlib.pyplot as plt

    fig_plot, axs_plot = plt.subplots(2, 3, figsize=(20, 9))
    fig_hist, axs_hist = plt.subplots(2, 3, figsize=(20, 9))

    for idx, att in enumerate(["sps", "das", "mat"]):
        # Plot
        axs_plot[0, idx].plot(data[att], c='b' if test_config["is_baseline"] else 'r')
        axs_plot[0, idx].set_title(att)
        # Hist
        axs_hist[0, idx].hist(data[att], bins=10, density=True, color='b' if test_config["is_baseline"] else 'r', alpha=0.6)
        axs_hist[0, idx].set_title(att)

    for idx, att in enumerate(["kw", "at", "dat"]):
        # Plot
        if test_config["is_baseline"]:
            axs_plot[1, idx].plot(data[att], c='b')
        axs_plot[1, idx].plot(data_pred[att], c='r')
        axs_plot[1, idx].set_title(att)
        # Hist
        if test_config["is_baseline"]:
            bins = np.linspace(min(min(data[att]), min(data_pred[att])), max(max(data[att]), max(data_pred[att])), 15)
            axs_hist[1, idx].hist(data[att], bins=bins, density=True, color='b', alpha=0.6)
        else:
            bins = np.linspace(min(data_pred[att]), max(data_pred[att]), 15)
        axs_hist[1, idx].hist(data_pred[att], bins=bins, density=True, color='r', alpha=0.6)
        axs_plot[1, idx].set_title(att)

    fig_plot.savefig(os.path.join(test_config["image_dir"], f"{test_config['image_prefix']}_plot.png"))
    plt.close(fig_plot)

    fig_hist.savefig(os.path.join(test_config["image_dir"], f"{test_config['image_prefix']}_hist.png"))
    plt.close(fig_hist)


if __name__ == "__main__":
    # train()

    test()
