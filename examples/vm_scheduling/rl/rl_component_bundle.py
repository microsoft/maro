# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from functools import partial
from typing import Any, Callable, Dict, Optional

from maro.rl.policy import AbsPolicy
from maro.rl.rl_component.rl_component_bundle import RLComponentBundle
from maro.rl.rollout import AbsEnvSampler
from maro.rl.training import AbsTrainer

from examples.vm_scheduling.rl.algorithms.ac import get_ac, get_ac_policy
from examples.vm_scheduling.rl.algorithms.dqn import get_dqn, get_dqn_policy
from examples.vm_scheduling.rl.config import algorithm, env_conf, num_features, num_pms, state_dim, test_env_conf
from examples.vm_scheduling.rl.env_sampler import VMEnvSampler


class VMBundle(RLComponentBundle):
    def get_env_config(self) -> dict:
        return env_conf

    def get_test_env_config(self) -> Optional[dict]:
        return test_env_conf

    def get_env_sampler(self) -> AbsEnvSampler:
        return VMEnvSampler(self.env, self.test_env)

    def get_agent2policy(self) -> Dict[Any, str]:
        return {"AGENT": f"{algorithm}.policy"}

    def get_policy_creator(self) -> Dict[str, Callable[[], AbsPolicy]]:
        action_num = num_pms + 1  # action could be any PM or postponement, hence the plus 1

        if algorithm == "ac":
            policy_creator = {
                f"{algorithm}.policy": partial(
                    get_ac_policy,
                    state_dim,
                    action_num,
                    num_features,
                    f"{algorithm}.policy",
                ),
            }
        elif algorithm == "dqn":
            policy_creator = {
                f"{algorithm}.policy": partial(
                    get_dqn_policy,
                    state_dim,
                    action_num,
                    num_features,
                    f"{algorithm}.policy",
                ),
            }
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        return policy_creator

    def get_trainer_creator(self) -> Dict[str, Callable[[], AbsTrainer]]:
        if algorithm == "ac":
            trainer_creator = {algorithm: partial(get_ac, state_dim, num_features, algorithm)}
        elif algorithm == "dqn":
            trainer_creator = {algorithm: partial(get_dqn, algorithm)}
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        return trainer_creator
