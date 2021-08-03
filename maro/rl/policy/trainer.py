# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
from multiprocessing.connection import Connection
from os import getcwd
from typing import Callable, Dict

from maro.communication import Proxy
from maro.rl.utils import MsgKey, MsgTag
from maro.utils import Logger


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
    policy_dict = {policy_name: func() for policy_name, func in create_policy_func_dict.items()}
    logger = Logger("TRAINER", dump_folder=log_dir)
    for name, state in initial_policy_states.items():
        policy_dict[name].set_state(state)
        logger.info(f"{trainer_id} initialized policy {name}")

    while True:
        msg = conn.recv()
        if msg["type"] == "train":
            t0 = time.time()
            for name, exp in msg["experiences"].items():
                policy_dict[name].store(exp)
                policy_dict[name].learn()
            logger.debug(f"total policy update time: {time.time() - t0}")
            conn.send({
                "policy": {name: policy_dict[name].get_state() for name in msg["experiences"]},
                "tracker": {name: policy_dict[name].tracker for name in msg["experiences"]}
            })
        elif msg["type"] == "quit":
            break


def trainer_node(
    group: str,
    trainer_idx: int,
    create_policy_func_dict: Dict[str, Callable],
    proxy_kwargs: dict = {},
    log_dir: str = getcwd()
):
    """Policy trainer process that can be launched on separate computation nodes.

    Args:
        group (str): Group name for the training cluster, which includes all trainers and a training manager that
            manages them.
        trainer_idx (int): Integer trainer index. The trainer's ID in the cluster will be "TRAINER.{trainer_idx}".
        create_policy_func_dict (dict): A dictionary mapping policy names to functions that create them. The policy
            creation function should have exactly one parameter which is the policy name and return an ``AbsPolicy``
            instance.
        proxy_kwargs: Keyword parameters for the internal ``Proxy`` instance. See ``Proxy`` class
            for details. Defaults to the empty dictionary.
        log_dir (str): Directory to store logs in. Defaults to the current working directory.
    """
    policy_dict = {}
    proxy = Proxy(group, "trainer", {"policy_manager": 1}, component_name=f"TRAINER.{trainer_idx}", **proxy_kwargs)
    logger = Logger(proxy.name, dump_folder=log_dir)

    for msg in proxy.receive():
        if msg.tag == MsgTag.EXIT:
            logger.info("Exiting...")
            proxy.close()
            break

        if msg.tag == MsgTag.INIT_POLICY_STATE:
            for name, state in msg.body[MsgKey.POLICY_STATE].items():
                policy_dict[name] = create_policy_func_dict[name]()
                policy_dict[name].set_state(state)
                logger.info(f"{proxy.name} initialized policy {name}")
            proxy.reply(msg, tag=MsgTag.INIT_POLICY_STATE_DONE)
        elif msg.tag == MsgTag.LEARN:
            t0 = time.time()
            for name, exp in msg.body[MsgKey.EXPERIENCES].items():
                policy_dict[name].store(exp)
                policy_dict[name].learn()

            msg_body = {
                MsgKey.POLICY_STATE: {name: policy_dict[name].get_state() for name in msg.body[MsgKey.EXPERIENCES]},
                MsgKey.TRACKER: {name: policy_dict[name].tracker for name in msg.body[MsgKey.EXPERIENCES]}
            }
            logger.debug(f"total policy update time: {time.time() - t0}")
            proxy.reply(msg, body=msg_body)
        elif msg.tag == MsgTag.SCATTER_EXP:
            t0 = time.time()
            msg_body = {
                MsgKey.TRACKER: {}
            }

            for name, exp in msg.body[MsgKey.EXPERIENCES].items():
                policy_dict[name].store(exp)
                msg_body[MsgKey.TRACKER][name] = policy_dict[name].tracker

            logger.debug(f"total scatter experience time: {time.time() - t0}")
            proxy.reply(msg, body=msg_body)
        elif msg.tag == MsgTag.GET_GRAD:
            t0 = time.time()
            # message: loss of each trainer node
            msg_body = {
                MsgKey.GRAD: {},
                MsgKey.TRACKER: {}
            }

            for name, _ in msg.body[MsgKey.EXPERIENCES].items():
                # TODO: add for-loop of train_epochs
                # for _ in range(policy_dict[name].config.train_epochs):
                loss = policy_dict[name].get_loss()

                # Collect gradient
                loss.backward()
                grad_dict = {}
                for param_name, param in policy_dict[name].q_net.named_parameters():
                    grad_dict[param_name] = param.grad

                msg_body[MsgKey.GRAD][name] = grad_dict
                msg_body[MsgKey.TRACKER][name] = policy_dict[name].tracker

            logger.debug(f"single step of get_loss time: {time.time() - t0}")
            proxy.reply(msg, body=msg_body)
        elif msg.tag == MsgTag.UPDATE_POLICY_STATE:
            for name, state in msg.body[MsgKey.POLICY_STATE].items():
                policy_dict[name].set_state(state)
            proxy.reply(msg, tag=MsgTag.UPDATE_POLICY_STATE_DONE)
