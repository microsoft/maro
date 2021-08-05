# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
from collections import defaultdict
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
    num_epochs: Dict[str, int],
    reset_memory: Dict[str, bool] = defaultdict(lambda: False),
    log_dir: str = getcwd()
):
    """Policy trainer process which can be spawned by a ``MultiProcessPolicyManager``.

    Args:
        trainer_id (int): Integer trainer ID.
        conn (Connection): Connection end for exchanging messages with the manager process.
        create_policy_func_dict (dict): A dictionary mapping policy names to functions that create them. The policy
            creation function should have exactly one parameter which is the policy name and return an ``AbsPolicy``
            instance.
        initial_policy_states (dict): States with which to initialize the policies.
        num_epochs (Dict[str, int]): Number of learning epochs for each policy. This determine the number of
            times ``policy.learn()`` is called in each call to ``update``. Defaults to None, in which case the
            number of learning epochs will be set to 1 for each policy.
        reset_memory (Dict[str, bool]): A dictionary of flags indicating whether each policy's experience memory
            should be reset after it is updated. It may be necessary to set this to True for on-policy algorithms
            to ensure that the experiences to be learned from stay up to date. Defaults to False for each policy.
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
                for _ in range(num_epochs[name]):
                    policy_dict[name].update()
                if reset_memory[name]:
                    policy_dict[name].reset_memory()
            logger.debug(f"total policy update time: {time.time() - t0}")
            conn.send({
                "policy": {name: policy_dict[name].algorithm.get_state() for name in msg["experiences"]}
            })
        elif msg["type"] == "quit":
            break


def trainer_node(
    group: str,
    trainer_idx: int,
    create_policy_func_dict: Dict[str, Callable],
    num_epochs: Dict[str, int],
    reset_memory: Dict[str, int] = defaultdict(lambda: False),
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
        num_epochs (Dict[str, int]): Number of learning epochs for each policy. This determine the number of times
            ``policy.update()`` is called in each learning round. Defaults to None, in which case the number of
            learning epochs will be set to 1 for each policy.
        reset_memory (Dict[str, bool]): A dictionary of flags indicating whether each policy's experience memory
            should be reset after it is updated. It may be necessary to set this to True for on-policy algorithms
            to ensure that the experiences to be learned from stay up to date. Defaults to False for each policy.
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

        elif msg.tag == MsgTag.INIT_POLICY_STATE:
            for name, state in msg.body[MsgKey.POLICY_STATE].items():
                policy_dict[name] = create_policy_func_dict[name]()
                policy_dict[name].set_state(state)
                logger.info(f"{proxy.name} initialized policy {name}")
            proxy.reply(msg, tag=MsgTag.INIT_POLICY_STATE_DONE)
        elif msg.tag == MsgTag.LEARN:
            t0 = time.time()
            for name, exp in msg.body[MsgKey.EXPERIENCES].items():
                policy_dict[name].store(exp)
                for _ in range(num_epochs[name]):
                    policy_dict[name].update()
                if reset_memory[name]:
                    policy_dict[name].reset_memory()

            msg_body = {
                MsgKey.POLICY_STATE:
                    {name: policy_dict[name].algorithm.get_state() for name in msg.body[MsgKey.EXPERIENCES]}
            }
            logger.info(f"total policy update time: {time.time() - t0}")
            proxy.reply(msg, tag=MsgTag.TRAIN_DONE, body=msg_body)
        elif msg.tag == MsgTag.GET_UPDATE_INFO:
            t0 = time.time()
            msg_body = {
                MsgKey.UPDATE_INFO:
                    {name: policy_dict[name].get_update_info(exp)
                        for name, exp in msg.body[MsgKey.EXPERIENCES].items()},
            }
            logger.info(f"total time to get update info: {time.time() - t0}")
            proxy.reply(msg, tag=MsgTag.UPDATE_INFO, body=msg_body)
        elif msg.tag == MsgTag.UPDATE_POLICY_STATE:
            for name, state in msg.body[MsgKey.POLICY_STATE].items():
                policy_dict[name].set_state(state)
                logger.info(f"{proxy.name} updated policy {name}")
