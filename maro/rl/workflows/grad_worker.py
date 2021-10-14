# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import importlib
import os
import sys
import time

from maro.communication import Proxy, SessionMessage
from maro.rl.utils import MsgKey, MsgTag
from maro.rl.workflows.helpers import from_env, get_default_log_dir
from maro.utils import Logger

sys.path.insert(0, from_env("SCENARIODIR"))
module = importlib.import_module(from_env("SCENARIO"))
policy_func_dict = getattr(module, "policy_func_dict")
log_dir = from_env("LOGDIR", required=False, default=get_default_log_dir(from_env("JOB")))
os.makedirs(log_dir, exist_ok=True)


if __name__ == "__main__":
    # TODO: WORKERID in docker compose script.
    worker_id = from_env("WORKERID")
    num_hosts = from_env("NUMHOSTS") if from_env("POLICYMANAGERTYPE") == "distributed" else 0
    max_cached_policies = from_env("MAXCACHED", required=False, default=10)

    group = os.getenv("POLICYGROUP", default="learn")
    policy_dict = {}
    active_policies = []
    if num_hosts == 0:
        # no remote nodes for policy hosts
        num_hosts = len(policy_func_dict)

    peers = {"policy_manager": 1, "policy_host": num_hosts, "task_queue": 1}
    proxy = Proxy(
        group, "grad_worker", peers, component_name=f"GRAD_WORKER.{worker_id}",
        redis_address=(from_env("REDISHOST"), from_env("REDISPORT")),
        max_peer_discovery_retries=50
    )
    logger = Logger(proxy.name, dump_folder=log_dir)

    for msg in proxy.receive():
        if msg.tag == MsgTag.EXIT:
            logger.info("Exiting...")
            proxy.close()
            break
        elif msg.tag == MsgTag.COMPUTE_GRAD:
            t0 = time.time()
            msg_body = {MsgKey.LOSS_INFO: dict(), MsgKey.POLICY_IDS: list()}
            for name, batch in msg.body[MsgKey.GRAD_TASK].items():
                if name not in policy_dict:
                    if len(policy_dict) > max_cached_policies:
                        # remove the oldest one when size exceeds.
                        policy_to_remove = active_policies.pop()
                        policy_dict.pop(policy_to_remove)
                    policy_dict[name] = policy_func_dict[name](name)
                    active_policies.insert(0, name)
                    logger.info(f"Initialized policies {name}")

                policy_dict[name].set_state(msg.body[MsgKey.POLICY_STATE][name])
                loss_info = policy_dict[name].get_batch_loss(batch, explicit_grad=True)
                msg_body[MsgKey.LOSS_INFO][name] = loss_info
                msg_body[MsgKey.POLICY_IDS].append(name)
                # put the latest one to queue head
                active_policies.remove(name)
                active_policies.insert(0, name)

            logger.debug(f"total policy update time: {time.time() - t0}")
            proxy.reply(msg, tag=MsgTag.COMPUTE_GRAD_DONE, body=msg_body)
            # release worker at task queue
            proxy.isend(SessionMessage(
                MsgTag.RELEASE_WORKER, proxy.name, "TASK_QUEUE", body={MsgKey.WORKER_ID: f"GRAD_WORKER.{worker_id}"}
            ))
        else:
            logger.info(f"Wrong message tag: {msg.tag}")
            raise TypeError
