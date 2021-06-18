# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
from multiprocessing.connection import Connection
from os import getcwd
from typing import Callable, Dict

from maro.communication import Proxy
from maro.utils import Logger

from .message_enums import MsgKey, MsgTag


def trainer_process(trainer_id: str, conn: Connection, create_policy_func_dict: Dict[str, Callable], log_dir: str):
    """Policy trainer process which can be spawned by a ``MultiProcessTrainingManager``.

    Args:
        trainer_id (str): Identifier for the trainer process for bookkeeping by the parent manager process.
        conn (Connection): Connection end for exchanging messages with the manager process.
        create_policy_func_dict (dict): A dictionary mapping policy names to functions that create them. The policy
            creation function should have exactly one parameter which is the policy name and return an ``AbsPolicy``
            instance.
        log_dir (str): Directory to store logs in. Defaults to the current working directory.
    """
    policy_dict = {policy_name: func(policy_name) for policy_name, func in create_policy_func_dict.items()}
    logger = Logger("TRAINER", dump_folder=log_dir)
    while True:
        msg = conn.recv()
        if msg["type"] == "train":
            t0 = time.time()
            updated = {
                name: policy_dict[name].get_state() for name, exp in msg["experiences"].items()
                if policy_dict[name].on_experiences(exp)
            }
            logger.debug(f"total policy update time: {time.time() - t0}")
            conn.send({"policy": updated})
        elif msg["type"] == "get_policy_state":
            policy_state_dict = {name: policy.get_state() for name, policy in policy_dict.items()}
            conn.send({"policy": policy_state_dict})
        elif msg["type"] == "quit":
            break


def trainer_node(
    trainer_id: str,
    create_policy_func_dict: Dict[str, Callable],
    group: str,
    log_dir: str = getcwd(),
    **proxy_kwargs
):
    """Policy trainer process that can be launched on separate computation nodes.

    Args:
        trainer_id (str): Identifier for the trainer process for bookkeeping by the parent manager process.
        create_policy_func_dict (dict): A dictionary mapping policy names to functions that create them. The policy
            creation function should have exactly one parameter which is the policy name and return an ``AbsPolicy``
            instance.
        group (str): Group name for the training cluster, which includes all trainers and a training manager that
            manages them.
        log_dir (str): Directory to store logs in. Defaults to the current working directory.
        proxy_kwargs: Keyword parameters for the internal ``Proxy`` instance. See ``Proxy`` class
            for details.
    """
    policy_dict = {policy_name: func() for policy_name, func in create_policy_func_dict.items()}
    proxy = Proxy(group, "trainer", {"training_manager": 1}, component_name=trainer_id, **proxy_kwargs)
    logger = Logger(proxy.name, dump_folder=log_dir)

    for msg in proxy.receive():
        if msg.tag == MsgTag.EXIT:
            logger.info("Exiting...")
            proxy.close()
            break

        if msg.tag == MsgTag.TRAIN:
            t0 = time.time()
            updated = {
                name: policy_dict[name].get_state() for name, exp in msg.body[MsgKey.EXPERIENCES].items()
                if policy_dict[name].on_experiences(exp)
            }
            logger.debug(f"total policy update time: {time.time() - t0}")
            proxy.reply(msg, body={MsgKey.POLICY: updated})
        elif msg.tag == MsgTag.GET_POLICY_STATE:
            policy_state_dict = {name: policy.get_state() for name, policy in policy_dict.items()}
            proxy.reply(msg, tag=MsgTag.POLICY_STATE, body={MsgKey.POLICY: policy_state_dict})
