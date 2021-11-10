# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from maro.rl.learning import DistributedPolicyManager, MultiProcessPolicyManager, SimplePolicyManager
from maro.rl.workflows.helpers import from_env, get_logger, get_scenario_module


def get_policy_manager():
    policy_func_dict = getattr(get_scenario_module(from_env("SCENARIODIR")), "policy_func_dict")
    manager_type = from_env("POLICYMANAGERTYPE")
    data_parallelism = from_env("DATAPARALLELISM", required=False, default=1)
    load_policy_dir = from_env("LOADDIR", required=False, default=None)
    group = from_env("POLICYGROUP", required=False, default="learn") if data_parallelism > 1 else None
    proxy_kwargs = {
        "redis_address": (from_env("REDISHOST"), from_env("REDISPORT")),
        "max_peer_discovery_retries": 50
    }
    checkpoint_dir = from_env("CHECKPOINTDIR", required=False, default=None)
    logger = get_logger(from_env("LOGDIR", required=False, default=os.getcwd()), from_env("JOB"), "POLICY_MANAGER")
    if manager_type == "simple":
        return SimplePolicyManager(
            policy_func_dict,
            load_dir=load_policy_dir,
            checkpoint_dir=checkpoint_dir,
            data_parallelism=data_parallelism,
            group=group,
            proxy_kwargs=proxy_kwargs,
            logger=logger
        )
    elif manager_type == "multi-process":
        return MultiProcessPolicyManager(
            policy_func_dict,
            load_dir=load_policy_dir,
            checkpoint_dir=checkpoint_dir,
            data_parallelism=data_parallelism,
            group=group,
            proxy_kwargs=proxy_kwargs,
            logger=logger
        )
    elif manager_type == "distributed":
        return DistributedPolicyManager(
            list(policy_func_dict.keys()), from_env("NUMHOSTS"),
            group=from_env("POLICYGROUP"),
            data_parallelism=data_parallelism,
            proxy_kwargs=proxy_kwargs,
            logger=logger
        )

    raise ValueError(
        f"Unsupported policy manager type: {manager_type}. Supported modes: simple, multi-process, distributed"
    )


if __name__ == "__main__":
    policy_manager = get_policy_manager()
    policy_manager.server(
        from_env("GROUP"),
        from_env("NUMROLLOUTS"),
        max_lag=from_env("MAXLAG", required=False, default=0),
        proxy_kwargs={
            "redis_address": (from_env("REDISHOST"), from_env("REDISPORT")),
            "max_peer_discovery_retries": 50
        }
    )
