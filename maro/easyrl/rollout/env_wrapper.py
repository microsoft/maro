# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import collections
from typing import Any, Dict, List

from maro.easyrl.policy import EasyPolicy
from maro.rl.rollout import AbsEnvSampler, ExpElement


class SimpleEasyRLJob(object):
    def __init__(self, env_sampler: AbsEnvSampler, agent_policy_dict: Dict[Any, EasyPolicy]) -> None:
        super(SimpleEasyRLJob, self).__init__()

        self._env_sampler = env_sampler
        self._env_sampler.build_easy(agent2policy={agent: policy.actor for agent, policy in agent_policy_dict.items()})
        self._agent_policy_dict = agent_policy_dict

    def train(self, num_epoch: int) -> None:
        for ep in range(num_epoch):
            exps_by_agent = self.sample(ep)
            for agent, policy in self._agent_policy_dict.items():
                policy.train_with_experiences(exps_by_agent[agent])

    def sample(self, ep: int) -> Dict[Any, Dict[int, List[ExpElement]]]:
        result = self._env_sampler.sample()
        exps_multi_env: List[List[ExpElement]] = result["experiences"]
        info_list: List[dict] = result["info"]

        self._env_sampler.post_collect(info_list, ep)

        ret: Dict[Any, Dict[int, List[ExpElement]]] = collections.defaultdict(lambda: collections.defaultdict(list))
        for env_idx, env_exps in enumerate(exps_multi_env):
            for element in env_exps:
                exp_by_agent = element.split_contents_by_agent()
                for agent, exp in exp_by_agent.items():
                    ret[agent][env_idx].append(exp)
        return ret
