# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.rl.rollout import RolloutProxy
from maro.rl.utils.common import from_env_as_int

if __name__ == "__main__":
    batch_proxy = RolloutProxy(
        from_env_as_int("NUM_ROLLOUT_WORKERS"),
        from_env_as_int("ROLLOUT_PROXY_FRONTEND_PORT"),
        from_env_as_int("ROLLOUT_PROXY_BACKEND_PORT")
    )
    batch_proxy.start()
