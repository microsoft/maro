# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
from os import getcwd
from typing import Callable, Dict

from maro.communication import Proxy
from maro.rl.policy import LossInfo, RLPolicy
from maro.rl.typing import Trajectory
from maro.rl.utils import MsgKey, MsgTag
from maro.utils import Logger


def policy_host(
    create_policy_func_dict: Dict[str, Callable[[str], RLPolicy]],
    host_idx: int,
    group: str,
    proxy_kwargs: dict = {},
    log_dir: str = getcwd()
):
    """Policy host process that can be launched on separate computation nodes.

    Args:
        create_policy_func_dict (dict): A dictionary mapping policy names to functions that create them. The policy
            creation function should have policy name as the only parameter and return an ``RLPolicy`` instance.
        host_idx (int): Integer host index. The host's ID in the cluster will be "POLICY_HOST.{host_idx}".
        group (str): Group name for the training cluster, which includes all policy hosts and a policy manager that
            manages them.
        proxy_kwargs: Keyword parameters for the internal ``Proxy`` instance. See ``Proxy`` class
            for details. Defaults to the empty dictionary.
        log_dir (str): Directory to store logs in. Defaults to the current working directory.
    """
    policy_dict = {}
    proxy = Proxy(group, "policy_host", {"policy_manager": 1}, component_name=f"POLICY_HOST.{host_idx}", **proxy_kwargs)
    logger = Logger(proxy.name, dump_folder=log_dir)

    for msg in proxy.receive():
        if msg.tag == MsgTag.EXIT:
            logger.info("Exiting...")
            proxy.close()
            break

        if msg.tag == MsgTag.INIT_POLICIES:
            for name in msg.body[MsgKey.POLICY_NAMES]:
                policy_dict[name] = create_policy_func_dict[name](name)

            proxy.reply(
                msg,
                tag=MsgTag.INIT_POLICIES_DONE,
                body={MsgKey.POLICY_STATE: {name: policy.get_state() for name, policy in policy_dict.items()}}
            )
        elif msg.tag == MsgTag.LEARN:
            t0 = time.time()
            for name, info_list in msg.body[MsgKey.ROLLOUT_INFO].items():
                if isinstance(info_list[0], Trajectory):
                    policy_dict[name].learn_from_multi_trajectories(info_list)
                elif isinstance(info_list[0], LossInfo):
                    policy_dict[name].apply(info_list)
                else:
                    raise TypeError(
                        f"Roll-out information must be of type 'Trajectory' or 'LossInfo', "
                        f"got {type(info_list[0])}"
                    )
            msg_body = {
                MsgKey.POLICY_STATE: {name: policy_dict[name].get_state() for name in msg.body[MsgKey.TRAJECTORIES]}
            }
            logger.debug(f"total policy update time: {time.time() - t0}")
            proxy.reply(msg, tag=MsgTag.LEARN_DONE, body=msg_body)
