# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.rl_v3.rollout import RolloutWorker
from maro.rl_v3.utils.common import from_env, from_env_as_int, get_module

if __name__ == "__main__":
    scenario = get_module(str(from_env("SCENARIO_PATH")))
    env_sampler_creator = getattr(scenario, "env_sampler_creator")
    worker = RolloutWorker(
        from_env_as_int("ID"),
        env_sampler_creator,
        str(from_env("ROLLOUT_PROXY_HOST")),
        router_port=from_env_as_int("ROLLOUT_PROXY_BACKEND_PORT")
    )
    worker.start()
