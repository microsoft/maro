from functools import partial
from typing import Any, Callable, Dict, Optional

from examples.cim.rl.config import action_num, algorithm, env_conf, num_agents, state_dim
from examples.cim.rl.env_sampler import CIMEnvSampler
from maro.rl.policy import AbsPolicy
from maro.rl.rl_component.rl_component_bundle import RLComponentBundle
from maro.rl.rollout import AbsEnvSampler
from maro.rl.training import AbsTrainer

from .algorithms.ac import get_ac_policy
from .algorithms.dqn import get_dqn_policy
from .algorithms.maddpg import get_maddpg_policy
from .algorithms.ppo import get_ppo_policy
from .algorithms.ac import get_ac
from .algorithms.ppo import get_ppo
from .algorithms.dqn import get_dqn
from .algorithms.maddpg import get_maddpg


class CIMBundle(RLComponentBundle):
    def get_env_config(self) -> dict:
        return env_conf

    def get_test_env_config(self) -> Optional[dict]:
        return None

    def get_env_sampler(self) -> AbsEnvSampler:
        return CIMEnvSampler()

    def get_agent2policy(self) -> Dict[Any, str]:
        return {agent: f"{algorithm}_{agent}.policy"for agent in self.env.agent_idx_list}

    def get_policy_creator(self) -> Dict[str, Callable[[], AbsPolicy]]:
        if algorithm == "ac":
            policy_creator = {
                f"{algorithm}_{i}.policy": partial(get_ac_policy, state_dim, action_num, f"{algorithm}_{i}.policy")
                for i in range(num_agents)
            }
        elif algorithm == "ppo":
            policy_creator = {
                f"{algorithm}_{i}.policy": partial(get_ppo_policy, state_dim, action_num, f"{algorithm}_{i}.policy")
                for i in range(num_agents)
            }
        elif algorithm == "dqn":
            policy_creator = {
                f"{algorithm}_{i}.policy": partial(get_dqn_policy, state_dim, action_num, f"{algorithm}_{i}.policy")
                for i in range(num_agents)
            }
        elif algorithm == "discrete_maddpg":
            policy_creator = {
                f"{algorithm}_{i}.policy": partial(get_maddpg_policy, state_dim, action_num, f"{algorithm}_{i}.policy")
                for i in range(num_agents)
            }
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        return policy_creator

    def get_trainer_creator(self) -> Dict[str, Callable[[], AbsTrainer]]:
        if algorithm == "ac":
            trainer_creator = {
                f"{algorithm}_{i}": partial(get_ac, state_dim, f"{algorithm}_{i}")
                for i in range(num_agents)
            }
        elif algorithm == "ppo":
            trainer_creator = {
                f"{algorithm}_{i}": partial(get_ppo, state_dim, f"{algorithm}_{i}")
                for i in range(num_agents)
            }
        elif algorithm == "dqn":
            trainer_creator = {
                f"{algorithm}_{i}": partial(get_dqn, f"{algorithm}_{i}")
                for i in range(num_agents)
            }
        elif algorithm == "discrete_maddpg":
            trainer_creator = {
                f"{algorithm}_{i}": partial(get_maddpg, state_dim, [1], f"{algorithm}_{i}")
                for i in range(num_agents)
            }
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        return trainer_creator

    def post_collect(self, info_list: list, ep: int, segment: int) -> None:
        # print the env metric from each rollout worker
        for info in info_list:
            print(f"env summary (episode {ep}, segment {segment}): {info['env_metric']}")

        # print the average env metric
        if len(info_list) > 1:
            metric_keys, num_envs = info_list[0]["env_metric"].keys(), len(info_list)
            avg_metric = {key: sum(info["env_metric"][key] for info in info_list) / num_envs for key in metric_keys}
            print(f"average env summary (episode {ep}, segment {segment}): {avg_metric}")

    def post_evaluate(self, info_list: list, ep: int) -> None:
        # print the env metric from each rollout worker
        for info in info_list:
            print(f"env summary (episode {ep}): {info['env_metric']}")

        # print the average env metric
        if len(info_list) > 1:
            metric_keys, num_envs = info_list[0]["env_metric"].keys(), len(info_list)
            avg_metric = {key: sum(info["env_metric"][key] for info in info_list) / num_envs for key in metric_keys}
            print(f"average env summary (episode {ep}): {avg_metric}")
