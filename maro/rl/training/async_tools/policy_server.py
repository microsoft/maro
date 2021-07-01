# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from os import getcwd

from maro.communication import Proxy
from maro.utils import Logger

from ..message_enums import MsgKey, MsgTag
from ..policy_manager import AbsPolicyManager


def policy_server(
    group: str,
    policy_manager: AbsPolicyManager,
    num_actors: int,
    max_lag: int = 0,
    proxy_kwargs: dict = {},
    log_dir: str = getcwd()
):
    """Policy server process.

    The process serves the latest policy states to a set of remote actors and receives simulated experiences from them.

    Args:
        group (str): Group name for the cluster that includes the server and all actors.
        policy_manager (AbsPolicyManager): An ``AbsPolicyManager`` instance that hosts all policies and updates
            them using experiences collected by the actors.
        num_actors (int): Number of remote actors to collect simulation experiences.
        max_lag (int): Maximum policy version lag allowed for experiences collected from remote roll-out workers.
            Experiences collected using policy versions older than (current_version - max_lag) will be discarded.
            Defaults to 0, in which case only experiences collected using the latest policy version will be returned.
        proxy_kwargs: Keyword parameters for the internal ``Proxy`` instance. See ``Proxy`` class
            for details. Defaults to the empty dictionary.
        log_dir (str): Directory to store logs in. Defaults to the current working directory.
    """
    peers = {"actor": num_actors}
    name = "POLICY_SERVER"
    proxy = Proxy(group, "policy_server", peers, component_name=name, **proxy_kwargs)
    logger = Logger(name, dump_folder=log_dir)

    num_active_actors = num_actors
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
        elif msg.tag == MsgTag.DONE:
            num_active_actors -= 1
            if num_active_actors == 0:
                return
