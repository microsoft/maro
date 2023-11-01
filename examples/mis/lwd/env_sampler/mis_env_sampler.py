# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from maro.rl.policy.abs_policy import AbsPolicy
from maro.rl.rollout import SimpleAgentWrapper, AbsEnvSampler, CacheElement
from maro.rl.utils import ndarray_to_tensor
from maro.rl.workflows.callback import Callback
from maro.simulator.core import Env

from examples.mis.lwd.env_sampler.baseline import greedy_mis_solver, uniform_mis_solver
from examples.mis.lwd.ppo.ppo import GraphBasedPPOPolicy
from examples.mis.lwd.ppo.replay_memory import GraphBasedExpElement
from examples.mis.lwd.simulator import Action, MISDecisionPayload, MISEnvMetrics, MISBusinessEngine


class MISAgentWrapper(SimpleAgentWrapper):
    def __init__(self, policy_dict: Dict[str, AbsPolicy], agent2policy: Dict[Any, str]) -> None:
        super().__init__(policy_dict, agent2policy)

    def _choose_actions_impl(self, state_by_agent: Dict[Any, torch.Tensor]) -> Dict[Any, np.ndarray]:
        assert len(state_by_agent) == 1
        for agent_name, state in state_by_agent.items():
            break

        policy_name = self._agent2policy[agent_name]
        policy = self._policy_dict[policy_name]
        assert isinstance(policy, GraphBasedPPOPolicy)

        assert isinstance(state, Tuple)
        assert len(state) == 2
        action = policy.get_actions(state[0], graph=state[1])
        return {agent_name: action}


class MISEnvSampler(AbsEnvSampler):
    def __init__(
        self,
        learn_env: Env,
        test_env: Env,
        policies: List[AbsPolicy],
        agent2policy: Dict[Any, str],
        trainable_policies: List[str] = None,
        reward_eval_delay: int = None,
        max_episode_length: int = None,
        diversity_reward_coef: float = 0.1,
        reward_normalization_base: float = None
    ) -> None:
        super(MISEnvSampler, self).__init__(
            learn_env=learn_env,
            test_env=test_env,
            policies=policies,
            agent2policy=agent2policy,
            trainable_policies=trainable_policies,
            agent_wrapper_cls=MISAgentWrapper,
            reward_eval_delay=reward_eval_delay,
            max_episode_length=max_episode_length,
        )
        be = learn_env.business_engine
        assert isinstance(be, MISBusinessEngine)
        self._device = be._device
        self._diversity_reward_coef = diversity_reward_coef
        self._reward_normalization_base = reward_normalization_base

        self._sample_metrics: List[tuple] = []
        self._eval_metrics: List[tuple] = []

    def _get_global_and_agent_state(
        self,
        event: Any,
        tick: int = None,
    ) -> Tuple[Optional[Any], Dict[Any, Union[np.ndarray, list]]]:
        return self._get_global_and_agent_state_impl(event, tick)

    def _get_global_and_agent_state_impl(
        self,
        event: MISDecisionPayload,
        tick: int = None,
    ) -> Tuple[Union[None, np.ndarray, list], Dict[Any, Union[np.ndarray, list]]]:
        vertex_states = event.vertex_states.unsqueeze(2).float().cpu()
        normalized_tick = torch.full(vertex_states.size(), tick / self.env._business_engine._max_tick)
        state = torch.cat([vertex_states, normalized_tick], dim=2).cpu().detach().numpy()
        return None, {0: (state, event.graph)}

    def _translate_to_env_action(self, action_dict: dict, event: Any) -> dict:
        return {k: Action(vertex_states=ndarray_to_tensor(v, self._device)) for k, v in action_dict.items()}

    def _get_reward(self, env_action_dict: dict, event: Any, tick: int) -> Dict[Any, float]:
        be = self.env.business_engine
        assert isinstance(be, MISBusinessEngine)

        cardinality_record_dict = self.env.metrics[MISEnvMetrics.IncludedNodeCount]
        hamming_distance_record_dict = self.env.metrics[MISEnvMetrics.HammingDistanceAmongSamples]

        assert (tick - 1) in cardinality_record_dict
        cardinality_reward = cardinality_record_dict[tick - 1] - cardinality_record_dict.get(tick - 2, 0)

        assert (tick - 1) in hamming_distance_record_dict
        diversity_reward = hamming_distance_record_dict[tick - 1]

        reward = cardinality_reward + self._diversity_reward_coef * diversity_reward
        reward /= self._reward_normalization_base

        return {0: reward.cpu().detach().numpy()}

    def _eval_baseline(self) -> Dict[str, float]:
        assert isinstance(self.env.business_engine, MISBusinessEngine)
        graph_list: List[Dict[int, List[int]]] = self.env.business_engine._batch_adjs
        num_samples: int = self.env.business_engine._num_samples

        def best_among_samples(
            solver: Callable[[Dict[int, List[int]]], List[int]],
            num_samples: int,
            graph: Dict[int, List[int]],
        ) -> int:
            res = 0
            for _ in range(num_samples):
                res = max(res, len(solver(graph)))
            return res

        graph_size_list = [len(graph) for graph in graph_list]
        uniform_size_list = [best_among_samples(uniform_mis_solver, num_samples, graph) for graph in graph_list]
        greedy_size_list = [best_among_samples(greedy_mis_solver, num_samples, graph) for graph in graph_list]

        return {
            "graph_size": np.mean(graph_size_list),
            "uniform_size": np.mean(uniform_size_list),
            "greedy_size": np.mean(greedy_size_list),
        }

    def sample(self, policy_state: Optional[Dict[str, Dict[str, Any]]] = None, num_steps: Optional[int] = None) -> dict:
        if policy_state is not None:  # Update policy state if necessary
            self.set_policy_state(policy_state)
        self._switch_env(self._learn_env)  # Init the env
        self._agent_wrapper.explore()  # Collect experience

        # One complete episode in one sample call here.
        self._reset()
        experiences: List[GraphBasedExpElement] = []

        while not self._end_of_episode:
            state = self._agent_state_dict[0][0]
            graph = self._agent_state_dict[0][1]
            # Get agent actions and translate them to env actions
            action_dict = self._agent_wrapper.choose_actions(self._agent_state_dict)
            env_action_dict = self._translate_to_env_action(action_dict, self._event)
            # Update env and get new states (global & agent)
            self._step(list(env_action_dict.values()))
            assert self._reward_eval_delay is None
            reward_dict = self._get_reward(env_action_dict, self._event, self.env.tick)

            experiences.append(
                GraphBasedExpElement(
                    state=state,
                    action=action_dict[0],
                    reward=reward_dict[0],
                    is_done=self.env.metrics[MISEnvMetrics.IsDoneMasks],
                    graph=graph,
                )
            )

            self._total_number_interactions += 1
            self._current_episode_length += 1
            self._post_step(None)

        return {
            "experiences": [experiences],
            "info": [],
        }

    def eval(self, policy_state: Dict[str, Dict[str, Any]] = None, num_episodes: int = 1) -> dict:
        self._switch_env(self._test_env)
        info_list = []

        for _ in range(num_episodes):
            self._reset()

            baseline_info = self._eval_baseline()
            info_list.append(baseline_info)

            if policy_state is not None:
                self.set_policy_state(policy_state)

            self._agent_wrapper.exploit()
            while not self._end_of_episode:
                action_dict = self._agent_wrapper.choose_actions(self._agent_state_dict)
                env_action_dict = self._translate_to_env_action(action_dict, self._event)
                # Update env and get new states (global & agent)
                self._step(list(env_action_dict.values()))

                self._post_eval_step(None)

        return {"info": info_list}

    def _post_step(self, cache_element: CacheElement) -> None:
        if not (self._end_of_episode or self.truncated):
            return

        node_count_record = self.env.metrics[MISEnvMetrics.IncludedNodeCount]
        num_steps = max(node_count_record.keys()) + 1
        # Report the mean among samples as the average MIS size in rollout.
        num_nodes = torch.mean(node_count_record[num_steps - 1]).item()

        self._sample_metrics.append((num_steps, num_nodes))

    def _post_eval_step(self, cache_element: CacheElement) -> None:
        if not (self._end_of_episode or self.truncated):
            return

        node_count_record = self.env.metrics[MISEnvMetrics.IncludedNodeCount]
        num_steps = max(node_count_record.keys()) + 1
        # Report the maximum among samples as the MIS size in evaluation.
        num_nodes = torch.mean(torch.max(node_count_record[num_steps - 1], dim=1).values).item()

        self._eval_metrics.append((num_steps, num_nodes))

    def post_collect(self, info_list: list, ep: int) -> None:
        assert len(self._sample_metrics) == 1, f"One Episode for One Rollout/Collection in Current Workflow Design."
        num_steps, mis_size = self._sample_metrics[0]

        cur = {
            "n_steps": num_steps,
            "avg_mis_size": mis_size,
            "n_interactions": self._total_number_interactions,
        }
        self.metrics.update(cur)

        # clear validation metrics
        self.metrics = {k: v for k, v in self.metrics.items() if not k.startswith("val/")}
        self._sample_metrics.clear()

    def post_evaluate(self, info_list: list, ep: int) -> None:
        num_eval = len(self._eval_metrics)
        assert num_eval > 0, f"Num evaluation rounds much be positive!"

        cur = {
            "val/num_eval": num_eval,
            "val/avg_n_steps": np.mean([n for n, _ in self._eval_metrics]),
            "val/avg_mis_size": np.mean([s for _, s in self._eval_metrics]),
            "val/std_mis_size": np.std([s for _, s in self._eval_metrics]),
            "val/avg_graph_size": np.mean([info["graph_size"] for info in info_list]),
            "val/uniform_size": np.mean([info["uniform_size"] for info in info_list]),
            "val/uniform_std": np.std([info["uniform_size"] for info in info_list]),
            "val/greedy_size": np.mean([info["greedy_size"] for info in info_list]),
            "val/greedy_std": np.std([info["greedy_size"] for info in info_list]),
        }
        self.metrics.update(cur)
        self._eval_metrics.clear()


class MISPlottingCallback(Callback):
    def __init__(self, log_dir: str) -> None:
        super().__init__()
        self._log_dir = log_dir

    def _plot_trainer(self) -> None:
        for trainer_name in self.training_manager._trainer_dict.keys():
            df = pd.read_csv(os.path.join(self._log_dir, f"{trainer_name}.csv"))
            columns = [
                "Steps",
                "Mean Reward",
                "Mean Return",
                "Mean Advantage",
                "0-Action",
                "1-Action",
                "2-Action",
                "Critic Loss",
                "Actor Loss",
            ]

            _, axes = plt.subplots(len(columns), sharex=False, figsize=(20, 21))

            for col, ax in zip(columns, axes):
                data = df[col].dropna().to_numpy()[:-1]
                ax.plot(data, label=col)
                ax.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(self._log_dir, f"{trainer_name}.png"))
            plt.close()
            plt.cla()
            plt.clf()

    @staticmethod
    def _plot_mean_std(
        ax: plt.Axes, x: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray, color: str, label: str,
    ) -> None:
        ax.plot(x, y_mean, label=label, color=color)
        ax.fill_between(x, y_mean - y_std, y_mean + y_std, color=color, alpha=0.2)

    def _plot_metrics(self) -> None:
        df_rollout = pd.read_csv(os.path.join(self._log_dir, "metrics_full.csv"))
        df_eval = pd.read_csv(os.path.join(self._log_dir, "metrics_valid.csv"))

        n_steps = df_rollout["n_steps"].to_numpy()
        eval_n_steps = df_eval["val/avg_n_steps"].to_numpy()

        eval_ep = df_eval["ep"].to_numpy()
        graph_size = df_eval["val/avg_graph_size"].to_numpy()

        mis_size = df_rollout["avg_mis_size"].to_numpy()
        eval_mis_size, eval_size_std = df_eval["val/avg_mis_size"].to_numpy(), df_eval["val/std_mis_size"].to_numpy()
        uniform_size, uniform_std = df_eval["val/uniform_size"].to_numpy(), df_eval["val/uniform_std"].to_numpy()
        greedy_size, greedy_std = df_eval["val/greedy_size"].to_numpy(), df_eval["val/greedy_std"].to_numpy()

        fig, (ax_t, ax_gsize, ax_mis) = plt.subplots(3, sharex=False, figsize=(20, 15))

        color_map = {
            "rollout": "cornflowerblue",
            "eval": "orange",
            "uniform": "green",
            "greedy": "firebrick",
        }

        ax_t.plot(n_steps, label="n_steps", color=color_map["rollout"])
        ax_t.plot(eval_ep, eval_n_steps, label="val/n_steps", color=color_map["eval"])

        ax_gsize.plot(eval_ep, graph_size, label="val/avg_graph_size", color=color_map["eval"])

        ax_mis.plot(mis_size, label="avg_mis_size", color=color_map["rollout"])
        self._plot_mean_std(ax_mis, eval_ep, eval_mis_size, eval_size_std, color_map["eval"], "val/mis_size")
        self._plot_mean_std(ax_mis, eval_ep, uniform_size, uniform_std, color_map["uniform"], "val/uniform_size")
        self._plot_mean_std(ax_mis, eval_ep, greedy_size, greedy_std, color_map["greedy"], "val/greedy_size")

        for ax in fig.get_axes():
            ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self._log_dir, "metrics.png"))
        plt.close()
        plt.cla()
        plt.clf()

    def on_validation_end(self, ep: int) -> None:
        self._plot_trainer()
        self._plot_metrics()
