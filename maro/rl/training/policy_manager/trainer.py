# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
from multiprocessing.connection import Connection
from os import getcwd
from typing import Callable, Dict

from maro.communication import Proxy
from maro.utils import Logger

from ..message_enums import MsgKey, MsgTag


def trainer_process(
    trainer_id: int,
    conn: Connection,
    create_policy_func_dict: Dict[str, Callable],
    initial_policy_states: dict,
    log_dir: str = getcwd()
):
    """Policy trainer process which can be spawned by a ``MultiProcessPolicyManager``.

    Args:
        trainer_id (int): Integer trainer ID.
        conn (Connection): Connection end for exchanging messages with the manager process.
        create_policy_func_dict (dict): A dictionary mapping policy names to functions that create them. The policy
            creation function should have exactly one parameter which is the policy name and return an ``AbsPolicy``
            instance.
        log_dir (str): Directory to store logs in. Defaults to the current working directory.
    """
    policy_dict = {policy_name: func(policy_name) for policy_name, func in create_policy_func_dict.items()}
    logger = Logger("TRAINER", dump_folder=log_dir)
    for name, state in initial_policy_states.items():
        policy_dict[name].set_state(state)
        logger.info(f"Trainer {trainer_id} initialized policy {name}")

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
    create_policy_func_dict: Dict[str, Callable],
    group: str,
    proxy_kwargs: dict = {},
    log_dir: str = getcwd()
):
    """Policy trainer process that can be launched on separate computation nodes.

    Args:
        create_policy_func_dict (dict): A dictionary mapping policy names to functions that create them. The policy
            creation function should have exactly one parameter which is the policy name and return an ``AbsPolicy``
            instance.
        group (str): Group name for the training cluster, which includes all trainers and a training manager that
            manages them.
        proxy_kwargs: Keyword parameters for the internal ``Proxy`` instance. See ``Proxy`` class
            for details. Defaults to the empty dictionary.
        log_dir (str): Directory to store logs in. Defaults to the current working directory.
    """
    policy_dict = {policy_name: func(policy_name) for policy_name, func in create_policy_func_dict.items()}
    proxy = Proxy(group, "trainer", {"policy_manager": 1}, **proxy_kwargs)
    logger = Logger(proxy.name, dump_folder=log_dir)

    for msg in proxy.receive():
        if msg.tag == MsgTag.EXIT:
            logger.info("Exiting...")
            proxy.close()
            break

        if msg.tag == MsgTag.INIT_POLICY_STATE:
            for name, state in msg.body[MsgKey.POLICY_STATE].items():
                policy_dict[name].set_state(state)
                logger.info(f"{proxy.name} initialized policy {name}")
            proxy.reply(msg, tag=MsgTag.INIT_POLICY_STATE_DONE)
        elif msg.tag == MsgTag.TRAIN:
            t0 = time.time()
            msg_body = {
                MsgKey.POLICY_STATE: {
                    name: policy_dict[name].get_state() for name, exp in msg.body[MsgKey.EXPERIENCES].items()
                    if policy_dict[name].on_experiences(exp)
                }
            }
            logger.debug(f"total policy update time: {time.time() - t0}")
            proxy.reply(msg, body=msg_body)
