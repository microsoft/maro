# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from os import getcwd
from typing import Dict, List

from maro.rl.policy import AbsPolicy
from maro.communication import Proxy, SessionType
from maro.utils import Logger


class InferenceManager:
    def __init__(
        self,
        policies: List[AbsPolicy],
        policy2server: Dict[str, str],
        group: str,
        log_dir: str = getcwd(),
        **proxy_kwargs
    ):
        self._logger = Logger("INFERENCE_MANAGER", dump_folder=log_dir)
        self.policy2server = policy2server
        self._names = list(self.policy2server.keys())
        peers = {"policy_server": len(set(self.policy2server.values()))}
        self._proxy = Proxy(group, "inference_manager", peers, **proxy_kwargs)
        self._policy_cache = policies

    def run(self):
        pass
