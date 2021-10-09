# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
from os import getcwd
from typing import Callable, Dict

from maro.communication import Proxy, SessionMessage
from maro.rl.utils import MsgKey, MsgTag
from maro.utils import Logger

# default group name for the cluster consisting of a policy manager and all policy hosts.
# If data parallelism is enabled, the gradient workers will also belong in this group.
DEFAULT_POLICY_GROUP = "policy_group_default"


def grad_worker(
    create_policy_func_dict: Dict[str, Callable],
    worker_idx: int,
    num_hosts: int,
    group: str = DEFAULT_POLICY_GROUP,
    proxy_kwargs: dict = {},
    max_policy_number: int = 10,
    log_dir: str = getcwd()
):
    """Stateless gradient workers that excute gradient computation tasks.

    Args:
        create_policy_func_dict (dict): A dictionary mapping policy names to functions that create them. The policy
            creation function should have policy name as the only parameter and return an ``RLPolicy`` instance.
        worker_idx (int): Integer worker index. The worker's ID in the cluster will be "GRAD_WORKER.{worker_idx}".
        num_hosts (int): Number of policy hosts, which is required to find peers in proxy initialization.
            num_hosts=0 means policy hosts are hosted by policy manager while no remote nodes for them.
        group (str): Group name for the training cluster, which includes all policy hosts and a policy manager that
            manages them.
        proxy_kwargs: Keyword parameters for the internal ``Proxy`` instance. See ``Proxy`` class
            for details. Defaults to an empty dictionary.
        max_policy_number (int): Maximum policy number in a single worker node. Defaults to 10.
        log_dir (str): Directory to store logs in. Defaults to the current working directory.
    """
    policy_dict = {}
    active_policies = []
    if num_hosts == 0:
        # no remote nodes for policy hosts
        num_hosts = len(create_policy_func_dict)
    str_id = f"GRAD_WORKER.{worker_idx}"
    peers = {"policy_manager": 1, "policy_host": num_hosts, "task_queue": 1}
    proxy = Proxy(group, "grad_worker", peers, component_name=str_id, **proxy_kwargs)
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
                    if len(policy_dict) > max_policy_number:
                        # remove the oldest one when size exceeds.
                        policy_to_remove = active_policies.pop()
                        policy_dict.pop(policy_to_remove)
                    policy_dict[name] = create_policy_func_dict[name](name)
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
                MsgTag.RELEASE_WORKER, proxy.name, "TASK_QUEUE", body={MsgKey.WORKER_ID: str_id}
            ))
        else:
            logger.info(f"Wrong message tag: {msg.tag}")
            raise TypeError
