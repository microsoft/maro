# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.rl.training.proxy import TrainingProxy
from maro.rl.utils.common import from_env_as_int

if __name__ == "__main__":
    proxy = TrainingProxy(
        frontend_port=from_env_as_int("TRAIN_PROXY_FRONTEND_PORT"),
        backend_port=from_env_as_int("TRAIN_PROXY_BACKEND_PORT")
    )
    proxy.start()
