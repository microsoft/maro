# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import time

from maro.communication import Proxy, SessionMessage
from maro.rl.utils import MsgKey, MsgTag
from maro.rl.workflows.helpers import from_env, get_logger, get_scenario_module

if __name__ == "__main__":
    # TODO: WORKERID in docker compose script.
    trainer_worker_func_dict = getattr(get_scenario_module(from_env("SCENARIODIR")), "trainer_worker_func_dict")
    worker_id = f"GRAD_WORKER.{from_env('WORKERID')}"
    num_trainer_workers = from_env("NUMTRAINERWORKERS") if from_env("TRAINERTYPE") == "distributed" else 0
    max_cached_policies = from_env("MAXCACHED", required=False, default=10)

    group = from_env("POLICYGROUP", required=False, default="learn")
    policy_dict = {}
    active_policies = []
    if num_trainer_workers == 0:
        # no remote nodes for trainer workers
        num_trainer_workers = len(trainer_worker_func_dict)

    peers = {"trainer": 1, "trainer_workers": num_trainer_workers, "task_queue": 1}
    proxy = Proxy(
        group, "grad_worker", peers, component_name=worker_id,
        redis_address=(from_env("REDISHOST"), from_env("REDISPORT")),
        max_peer_discovery_retries=50
    )
    logger = get_logger(from_env("LOGDIR", required=False, default=os.getcwd()), from_env("JOB"), worker_id)

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
                    # Initialize
                    policy_dict[name] = trainer_worker_func_dict[name](name)
                    active_policies.insert(0, name)
                    logger.info(f"Initialized policies {name}")

                policy_dict[name].set_trainer_state_dict(msg.body[MsgKey.POLICY_STATE][name])
                grad_dict = policy_dict[name].get_batch_grad(batch, scope=msg.body[MsgKey.GRAD_SCOPE][name])
                msg_body[MsgKey.LOSS_INFO][name] = grad_dict
                msg_body[MsgKey.POLICY_IDS].append(name)
                # put the latest one to queue head
                active_policies.remove(name)
                active_policies.insert(0, name)

            logger.debug(f"total policy update time: {time.time() - t0}")
            proxy.reply(msg, tag=MsgTag.COMPUTE_GRAD_DONE, body=msg_body)
            # release worker at task queue
            proxy.isend(SessionMessage(
                MsgTag.RELEASE_WORKER, proxy.name, "TASK_QUEUE", body={MsgKey.WORKER_ID: worker_id}
            ))
        else:
            logger.info(f"Wrong message tag: {msg.tag}")
            raise TypeError
