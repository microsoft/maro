# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
from os import getcwd
from typing import List

from maro.communication import Proxy
from maro.rl.policy import AbsPolicy
from maro.utils import Logger

from .message_enums import MsgKey, MsgTag


class PolicyServer:
    def __init__(
        self,
        policies: List[AbsPolicy],
        group: str,
        name: str,
        log_dir: str = getcwd(),
        **proxy_kwargs
    ):
        self.policy_dict = {policy.name: policy for policy in policies}
        self._proxy = Proxy(group, "policy_server", {"policy_manager": 1}, component_name=name, **proxy_kwargs)
        self._logger = Logger(self._proxy.name, dump_folder=log_dir)

    def run(self):
        for msg in self._proxy.receive():
            if msg.tag == MsgTag.EXIT:
                self._logger.info("Exiting...")
                self._proxy.close()
                break

            if msg.tag == MsgTag.TRAIN:
                t0 = time.time()
                updated = {
                    name: self.policy_dict[name].get_state() for name, exp in msg.body[MsgKey.EXPERIENCES].items()
                    if self.policy_dict[name].on_experiences(exp)
                }
                t1 = time.time()
                self._logger.debug(f"total policy update time: {t1 - t0}")
                self._proxy.reply(msg, body={MsgKey.POLICY: updated})
                self._logger.debug(f"reply time: {time.time() - t1}")
            elif msg.tag == MsgTag.GET_POLICY_STATE:
                policy_state_dict = {name: policy.get_state() for name, policy in self.policy_dict.items()}
                self._proxy.reply(msg, tag=MsgTag.POLICY_STATE, body={MsgKey.POLICY: policy_state_dict})
