# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from os import getcwd

from maro.communication import Proxy
from maro.utils import Logger

from ..message_enums import MsgKey, MsgTag
from ..policy_manager import AbsPolicyManager


def policy_server(
    policy_manager: AbsPolicyManager,
    num_actors: int,
    group: str,
    max_lag: int = 0,
    proxy_kwargs: dict = {},
    log_dir: str = getcwd()
):
    peers = {"actor": num_actors}
    proxy = Proxy(group, "policy_server", peers, **proxy_kwargs)
    logger = Logger("POLICY_SERVER", dump_folder=log_dir)

    for msg in proxy.receive():
        if msg.tag == MsgTag.GET_INITIAL_POLICY_STATE:
            proxy.reply(
                msg, tag=MsgTag.POLICY_STATE,
                body={MsgKey.POLICY_STATE: policy_manager.get_state(), MsgKey.VERSION: policy_manager.version}
            )
            policy_manager.reset_update_status()
        elif msg.tag == MsgTag.COLLECT_DONE:
            if policy_manager.version - msg.body[MsgKey.VERSION] > max_lag:
                logger.info(
                    f"Ignored a message because it contains experiences generated using a stale policy version. "
                    f"Expected experiences generated using policy versions no earlier than "
                    f"{policy_manager.version - max_lag}, got {msg.body[MsgKey.VERSION]}"
                )
            else:
                policy_manager.on_experiences(msg.body[MsgKey.EXPERIENCES])
            proxy.reply(
                msg, tag=MsgTag.POLICY_STATE,
                body={MsgKey.POLICY_STATE: policy_manager.get_state(), MsgKey.VERSION: policy_manager.version}
            )
            policy_manager.reset_update_status()
