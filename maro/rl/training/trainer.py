# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from os import getcwd
from typing import List

from maro.communication import Proxy
from maro.rl.policy import AbsPolicy
from maro.utils import Logger

from .message_enums import MsgKey, MsgTag


class Trainer:
    def __init__(
        self,
        policies: List[AbsPolicy],
        group: str,
        name: str,
        log_dir: str = getcwd(),
        **proxy_kwargs
    ):
        self.policy_dict = {policy.name: policy for policy in policies}
        self._proxy = Proxy(group, "trainer", {"policy_manager": 1}, component_name=name, **proxy_kwargs)
        self._logger = Logger(self._proxy.name, dump_folder=log_dir)
        self._updated = {policy_name: True for policy_name in self.policy_dict}

    def run(self):
        for msg in self._proxy.receive():
            if msg.tag == MsgTag.EXIT:
                self._logger.info("Exiting...")
                self._proxy.close()
                break

            if msg.tag == MsgTag.TRAIN:
                for name, exp in msg.body[MsgKey.EXPERIENCES].items():
                    self._updated[name] = self.policy_dict[name].on_experiences(exp)
            elif msg.tag == MsgTag.GET_POLICY_STATE:
                updated_policy_state_dict = {
                    name: policy.get_state() for name, policy in self.policy_dict.items() if self._updated[name]    
                }
                self._proxy.reply(msg, tag=MsgTag.POLICY_STATE, body={MsgKey.POLICY: updated_policy_state_dict})
                self._updated = {policy_name: False for policy_name in self.policy_dict}
