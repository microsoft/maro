# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from os import getcwd
from typing import Dict, List

from maro.communication import Proxy
from maro.rl.policy import AbsPolicy
from maro.utils import Logger

from .message_enums import MsgKey, MsgTag


class PolicyServerGateway:
    def __init__(
        self,
        policy2trainer: Dict[str, str],
        group: str,
        log_dir: str = getcwd(),
        **proxy_kwargs
    ):
        super().__init__()
        self._logger = Logger("POLICY_SERVER_GATEWAY", dump_folder=log_dir)
        self.policy2trainer = policy2trainer
        self._names = list(self.policy2trainer.keys())
        peers = {"policy_server": len(set(self.policy2trainer.values()))}
        self._proxy = Proxy(group, "policy_server_gateway", peers, **proxy_kwargs)

    def run(self):
        for msg in self._proxy.receive():
            if msg.tag == MsgTag.CHOOSE_ACTION:
                for name, exp in msg.body[MsgKey.EXPERIENCES].items():
                    self._updated[name] = self.policy_dict[name].on_experiences(exp)


class PolicyServer:
    def __init__(
        self,
        policies: List[AbsPolicy],
        group: str,
        name: str,
        as_server: bool = False,
        log_dir: str = getcwd(),
        **proxy_kwargs
    ):
        self.policy_dict = {policy.name: policy for policy in policies}
        peers = {"policy_manager": 1}
        if as_server:
            peers["policy_server_gateway"] = 1
        self._proxy = Proxy(group, "policy_server", peers, component_name=name, **proxy_kwargs)
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
            elif msg.tag == MsgTag.CHOOSE_ACTION:
                action_by_policy_name = {
                    policy_name: self.policy_dict[policy_name].choose_action(state_batch)
                    for policy_name, state_batch in msg.body[MsgKey.STATE].items()
                }
                self._proxy.reply(msg, tag=MsgTag.ACTION, body={MsgKey.ACTION: action_by_policy_name})  
