# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.rl.training import TrainingProxy
from maro.rl.utils.common import get_env, int_or_none

if __name__ == "__main__":
    proxy = TrainingProxy(
        frontend_port=int_or_none(get_env("TRAIN_PROXY_FRONTEND_PORT")),
        backend_port=int_or_none(get_env("TRAIN_PROXY_BACKEND_PORT")),
    )
    proxy.start()
