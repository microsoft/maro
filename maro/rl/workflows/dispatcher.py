# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.rl.training.dispatcher import TrainOpsDispatcher
from maro.rl.utils.common import from_env_as_int

if __name__ == "__main__":
    dispatcher = TrainOpsDispatcher(
        frontend_port=from_env_as_int("DISPATCHER_FRONTEND_PORT"),
        backend_port=from_env_as_int("DISPATCHER_BACKEND_PORT")
    )
    dispatcher.start()
