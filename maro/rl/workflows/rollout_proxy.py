# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.rl.rollout import RolloutDispatcher
from maro.rl.utils.common import from_env_as_int

if __name__ == "__main__":
    batch_proxy = RolloutDispatcher(
        num_workers=from_env_as_int("NUM_ROLLOUT_WORKERS"),
        frontend_port=from_env_as_int("ROLLOUT_PROXY_FRONTEND_PORT"),
        backend_port=from_env_as_int("ROLLOUT_PROXY_BACKEND_PORT")
    )
    batch_proxy.start()
