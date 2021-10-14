# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import importlib
import os
import sys
import time

from maro.communication import Proxy
from maro.rl.utils import MsgKey, MsgTag
from maro.rl.workflows.helpers import from_env, get_default_log_dir
from maro.utils import Logger

sys.path.insert(0, from_env("SCENARIODIR"))
module = importlib.import_module(from_env("SCENARIO"))
policy_func_dict = getattr(module, "policy_func_dict")

checkpoint_dir = from_env("CHECKPOINTDIR", required=False, default=None)
if checkpoint_dir:
    os.makedirs(checkpoint_dir, exist_ok=True)
load_policy_dir = from_env("LOADDIR", required=False, default=None)
log_dir = from_env("LOGDIR", required=False, default=get_default_log_dir(from_env("JOB")))
os.makedirs(log_dir, exist_ok=True)


if __name__ == "__main__":
    host_id = from_env("HOSTID")
    peers = {"policy_manager": 1}
    data_parallel = os.getenv("DATAPARALLEL") == "True"
    if data_parallel:
        num_grad_workers = from_env("NUMGRADWORKERS")
        peers["grad_worker"] = num_grad_workers
        peers["task_queue"] = 1

    if host_id is None:
        raise ValueError("missing environment variable: HOSTID")

    group = from_env("POLICYGROUP")
    policy_dict, checkpoint_path = {}, {}

    proxy = Proxy(
        group, "policy_host", peers,
        component_name=f"POLICY_HOST.{host_id}",
        redis_address=(from_env("REDISHOST"), from_env("REDISPORT")),
        max_peer_discovery_retries=50
    )
    logger = Logger(proxy.name, dump_folder=log_dir)

    for msg in proxy.receive():
        if msg.tag == MsgTag.EXIT:
            logger.info("Exiting...")
            proxy.close()
            break
        elif msg.tag == MsgTag.INIT_POLICIES:
            for id_ in msg.body[MsgKey.POLICY_IDS]:
                policy_dict[id_] = policy_func_dict[id_](id_)
                checkpoint_path[id_] = os.path.join(checkpoint_dir, id_) if checkpoint_dir else None
                if load_policy_dir:
                    path = os.path.join(load_policy_dir, id_)
                    if os.path.exists(path):
                        policy_dict[id_].load(path)
                        logger.info(f"Loaded policy {id_} from {path}")
                if data_parallel:
                    policy_dict[id_].data_parallel_with_existing_proxy(proxy)

            logger.info(f"Initialized policies {msg.body[MsgKey.POLICY_IDS]}")
            proxy.reply(
                msg,
                tag=MsgTag.INIT_POLICIES_DONE,
                body={MsgKey.POLICY_STATE: {id_: policy.get_state() for id_, policy in policy_dict.items()}}
            )
        elif msg.tag == MsgTag.LEARN:
            t0 = time.time()
            for id_, info in msg.body[MsgKey.ROLLOUT_INFO].items():
                # in some cases e.g. Actor-Critic that get loss from rollout workers
                if isinstance(info, list):
                    logger.info("updating with loss info")
                    policy_dict[id_].update(info)
                else:
                    if data_parallel:
                        logger.info("learning on remote grad workers")
                        policy_dict[id_].learn_with_data_parallel(info)
                    else:
                        logger.info("learning from batch")
                        policy_dict[id_].learn(info)

                if checkpoint_path[id_]:
                    policy_dict[id_].save(checkpoint_path[id_])
                    logger.info(f"Saved policy {id_} to {checkpoint_path[id_]}")

            msg_body = {
                MsgKey.POLICY_STATE: {name: policy_dict[name].get_state() for name in msg.body[MsgKey.ROLLOUT_INFO]}
            }
            logger.info(f"total policy update time: {time.time() - t0}")
            proxy.reply(msg, tag=MsgTag.LEARN_DONE, body=msg_body)
        else:
            logger.info(f"Wrong message tag: {msg.tag}")
            raise TypeError
