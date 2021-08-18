# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
from os import getcwd
from typing import Callable, Dict

from maro.communication import Proxy
from maro.rl.utils import MsgKey, MsgTag
from maro.utils import Logger


def worker(
    group: str,
    worker_idx: int,
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
    proxy = Proxy(
        group, "GRADWORKER", {"grad_manager": 1}, component_name=f"GRADWORKER.{worker_idx}", **proxy_kwargs
    )
    logger = Logger(proxy.name, dump_folder=log_dir)

    for msg in proxy.receive():
        if msg.tag == MsgTag.EXIT:
            logger.info("Exiting...")
            proxy.close()
            break

        if msg.tag == MsgTag.COMPUTE_GRAD:
            t0 = time.time()
            task = msg.body[MsgKey.GRAD_TASK]
            policy_name = task.policy_name
            if policy_name not in policy_dict:
                policy_dict[policy_name] = create_policy_func_dict[policy_name]()
            msg_body = {MsgKey.LOSS_INFO: policy_dict[policy_name].get_loss_info(task.batch, with_grad=True)}
            logger.debug(f"total policy update time: {time.time() - t0}")
            proxy.reply(msg, tag=MsgTag.COMPUTE_GRAD_DONE, body=msg_body)
