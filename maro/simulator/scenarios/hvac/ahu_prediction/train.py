# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from maro.simulator.scenarios.hvac.ahu_prediction.config import data_config, fcn_config, training_config
from maro.simulator.scenarios.hvac.ahu_prediction.data_processing import AhuDataset, generate_dataset
from maro.simulator.scenarios.hvac.ahu_prediction.model import FCN_model


def train():
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

    model = FCN_model(**fcn_config)
    optimizer = torch.optim.Adamax(
        params=model.parameters(),
        lr=training_config["learning_rate"],
    )

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
        print(f"epoch: {epoch:4}, train: {mean_loss:.3}, val: {mean_metric:.3}")

        if mean_metric < min_val:
            best_ep = epoch
            best_loss = mean_loss
            min_val = mean_metric

            torch.save(model.state_dict(), os.path.join(training_config["model_path"], "best_model.pt"))

        if epoch - best_ep >= training_config["early_stopping_patience"]:
            break

    print(f"Training finished, best model generated in epoch: {best_ep}")
    os.rename(
        src=os.path.join(training_config["model_path"], "best_model.pt"),
        dst=os.path.join(training_config["model_path"], f"best_model_ep{best_ep}_loss{best_loss:.3}_val{min_val:.3}.pt")
    )


if __name__ == "__main__":
    train()
