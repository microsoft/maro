import time
from functools import partial
from os import makedirs
from os.path import dirname, join, realpath
from typing import Any, Callable, Dict, Optional

from matplotlib import pyplot as plt

from examples.vm_scheduling.rl.algorithms.ac import get_ac_policy
from examples.vm_scheduling.rl.algorithms.dqn import get_dqn_policy
from examples.vm_scheduling.rl.config import algorithm, env_conf, num_features, num_pms, state_dim, test_env_conf
from examples.vm_scheduling.rl.env_sampler import VMEnvSampler
from maro.rl.policy import AbsPolicy
from maro.rl.rl_component.rl_component_bundle import RLComponentBundle
from maro.rl.rollout import AbsEnvSampler
from maro.rl.training import AbsTrainer

timestamp = str(time.time())
plt_path = join(dirname(realpath(__file__)), "plots", timestamp)
makedirs(plt_path, exist_ok=True)


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
                    get_ac_policy, state_dim, action_num, num_features, f"{algorithm}.policy",
                )
            }
        elif algorithm == "dqn":
            policy_creator = {
                f"{algorithm}.policy": partial(
                    get_dqn_policy, state_dim, action_num, num_features, f"{algorithm}.policy",
                )
            }
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        return policy_creator

    def get_trainer_creator(self) -> Dict[str, Callable[[], AbsTrainer]]:
        if algorithm == "ac":
            from .algorithms.ac import get_ac, get_ac_policy
            trainer_creator = {algorithm: partial(get_ac, state_dim, num_features, algorithm)}
        elif algorithm == "dqn":
            from .algorithms.dqn import get_dqn, get_dqn_policy
            trainer_creator = {algorithm: partial(get_dqn, algorithm)}
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
            avg_metric = {key: sum(tr["env_metric"][key] for tr in info_list) / num_envs for key in metric_keys}
            print(f"average env metric (episode {ep}, segment {segment}): {avg_metric}")

    def post_evaluate(self, info_list: list, ep: int) -> None:
        # print the env metric from each rollout worker
        for info in info_list:
            print(f"env summary (evaluation episode {ep}): {info['env_metric']}")

        # print the average env metric
        if len(info_list) > 1:
            metric_keys, num_envs = info_list[0]["env_metric"].keys(), len(info_list)
            avg_metric = {key: sum(tr["env_metric"][key] for tr in info_list) / num_envs for key in metric_keys}
            print(f"average env metric (evaluation episode {ep}): {avg_metric}")

        for info in info_list:
            core_requirement = info["actions_by_core_requirement"]
            action_sequence = info["action_sequence"]
            # plot action sequence
            fig = plt.figure(figsize=(40, 32))
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(action_sequence)
            fig.savefig(f"{plt_path}/action_sequence_{ep}")
            plt.cla()
            plt.close("all")

            # plot with legal action mask
            fig = plt.figure(figsize=(40, 32))
            for idx, key in enumerate(core_requirement.keys()):
                ax = fig.add_subplot(len(core_requirement.keys()), 1, idx + 1)
                for i in range(len(core_requirement[key])):
                    if i == 0:
                        ax.plot(core_requirement[key][i][0] * core_requirement[key][i][1], label=str(key))
                        ax.legend()
                    else:
                        ax.plot(core_requirement[key][i][0] * core_requirement[key][i][1])

            fig.savefig(f"{plt_path}/values_with_legal_action_{ep}")

            plt.cla()
            plt.close("all")

            # plot without legal actin mask
            fig = plt.figure(figsize=(40, 32))

            for idx, key in enumerate(core_requirement.keys()):
                ax = fig.add_subplot(len(core_requirement.keys()), 1, idx + 1)
                for i in range(len(core_requirement[key])):
                    if i == 0:
                        ax.plot(core_requirement[key][i][0], label=str(key))
                        ax.legend()
                    else:
                        ax.plot(core_requirement[key][i][0])

            fig.savefig(f"{plt_path}/values_without_legal_action_{ep}")

            plt.cla()
            plt.close("all")
