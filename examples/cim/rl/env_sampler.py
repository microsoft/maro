# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Dict, List, Tuple, Union

import numpy as np

from maro.rl.rollout import AbsEnvSampler, CacheElement
from maro.simulator.scenarios.cim.common import Action, ActionType, DecisionEvent

from .config import action_shaping_conf, port_attributes, reward_shaping_conf, state_shaping_conf, vessel_attributes


class CIMEnvSampler(AbsEnvSampler):
    def _get_global_and_agent_state_impl(
        self,
        event: DecisionEvent,
        tick: int = None,
    ) -> Tuple[Union[None, np.ndarray, List[object]], Dict[Any, Union[np.ndarray, List[object]]]]:
        tick = self._env.tick
        vessel_snapshots, port_snapshots = self._env.snapshot_list["vessels"], self._env.snapshot_list["ports"]
        port_idx, vessel_idx = event.port_idx, event.vessel_idx
        ticks = [max(0, tick - rt) for rt in range(state_shaping_conf["look_back"] - 1)]
        future_port_list = vessel_snapshots[tick:vessel_idx:"future_stop_list"].astype("int")
        state = np.concatenate(
            [
                port_snapshots[ticks : [port_idx] + list(future_port_list) : port_attributes],
                vessel_snapshots[tick:vessel_idx:vessel_attributes],
            ],
        )
        return state, {port_idx: state}

    def _translate_to_env_action(
        self,
        action_dict: Dict[Any, Union[np.ndarray, List[object]]],
        event: DecisionEvent,
    ) -> Dict[Any, object]:
        action_space = action_shaping_conf["action_space"]
        finite_vsl_space = action_shaping_conf["finite_vessel_space"]
        has_early_discharge = action_shaping_conf["has_early_discharge"]

        port_idx, model_action = list(action_dict.items()).pop()

        vsl_idx, action_scope = event.vessel_idx, event.action_scope
        vsl_snapshots = self._env.snapshot_list["vessels"]
        vsl_space = vsl_snapshots[self._env.tick : vsl_idx : vessel_attributes][2] if finite_vsl_space else float("inf")

        percent = abs(action_space[model_action[0]])
        zero_action_idx = len(action_space) / 2  # index corresponding to value zero.
        if model_action < zero_action_idx:
            action_type = ActionType.LOAD
            actual_action = min(round(percent * action_scope.load), vsl_space)
        elif model_action > zero_action_idx:
            action_type = ActionType.DISCHARGE
            early_discharge = (
                vsl_snapshots[self._env.tick : vsl_idx : "early_discharge"][0] if has_early_discharge else 0
            )
            plan_action = percent * (action_scope.discharge + early_discharge) - early_discharge
            actual_action = round(plan_action) if plan_action > 0 else round(percent * action_scope.discharge)
        else:
            actual_action, action_type = 0, None

        return {port_idx: Action(vsl_idx, int(port_idx), actual_action, action_type)}

    def _get_reward(self, env_action_dict: Dict[Any, object], event: DecisionEvent, tick: int) -> Dict[Any, float]:
        start_tick = tick + 1
        ticks = list(range(start_tick, start_tick + reward_shaping_conf["time_window"]))

        # Get the ports that took actions at the given tick
        ports = [int(port) for port in list(env_action_dict.keys())]
        port_snapshots = self._env.snapshot_list["ports"]
        future_fulfillment = port_snapshots[ticks:ports:"fulfillment"].reshape(len(ticks), -1)
        future_shortage = port_snapshots[ticks:ports:"shortage"].reshape(len(ticks), -1)

        decay_list = [reward_shaping_conf["time_decay"] ** i for i in range(reward_shaping_conf["time_window"])]
        rewards = np.float32(
            reward_shaping_conf["fulfillment_factor"] * np.dot(future_fulfillment.T, decay_list)
            - reward_shaping_conf["shortage_factor"] * np.dot(future_shortage.T, decay_list),
        )
        return {agent_id: reward for agent_id, reward in zip(ports, rewards)}

    def _post_step(self, cache_element: CacheElement) -> None:
        self._info["env_metric"] = self._env.metrics

    def _post_eval_step(self, cache_element: CacheElement) -> None:
        self._post_step(cache_element)

    def post_collect(self, ep: int) -> None:
        if len(self._info_list) == 0:
            return

        # print the env metric from each rollout worker
        for info in self._info_list:
            print(f"env summary (episode {ep}): {info['env_metric']}")

        # average env metric
        metric_keys, num_envs = self._info_list[0]["env_metric"].keys(), len(self._info_list)
        avg_metric = {key: sum(info["env_metric"][key] for info in self._info_list) / num_envs for key in metric_keys}
        avg_metric["n_episode"] = len(self._info_list)
        print(f"average env summary (episode {ep}): {avg_metric}")

        self.metrics.update(avg_metric)
        self.metrics = {k: v for k, v in self.metrics.items() if not k.startswith("val/")}
        self._info_list.clear()

    def post_evaluate(self, ep: int) -> None:
        if len(self._info_list) == 0:
            return

        # print the env metric from each rollout worker
        for info in self._info_list:
            print(f"env summary (episode {ep}): {info['env_metric']}")

        # average env metric
        metric_keys, num_envs = self._info_list[0]["env_metric"].keys(), len(self._info_list)
        avg_metric = {key: sum(info["env_metric"][key] for info in self._info_list) / num_envs for key in metric_keys}
        avg_metric["n_episode"] = len(self._info_list)
        print(f"average env summary (episode {ep}): {avg_metric}")

        self.metrics.update({"val/" + k: v for k, v in avg_metric.items()})
        self._info_list.clear()

    def monitor_metrics(self) -> float:
        return -self.metrics["val/shortage_percentage"]

    @staticmethod
    def merge_metrics(metrics_list: List[dict]) -> dict:
        n_episode = sum(m["n_episode"] for m in metrics_list)
        metrics: dict = {
            "order_requirements": sum(m["order_requirements"] * m["n_episode"] for m in metrics_list) / n_episode,
            "container_shortage": sum(m["container_shortage"] * m["n_episode"] for m in metrics_list) / n_episode,
            "operation_number": sum(m["operation_number"] * m["n_episode"] for m in metrics_list) / n_episode,
            "n_episode": n_episode,
        }
        metrics["shortage_percentage"] = metrics["container_shortage"] / metrics["order_requirements"]

        metrics_list = [m for m in metrics_list if "val/shortage_percentage" in m]
        if len(metrics_list) > 0:
            n_episode = sum(m["val/n_episode"] for m in metrics_list)
            metrics.update({
                "val/order_requirements": sum(m["val/order_requirements"] * m["val/n_episode"] for m in metrics_list) / n_episode,
                "val/container_shortage": sum(m["val/container_shortage"] * m["val/n_episode"] for m in metrics_list) / n_episode,
                "val/operation_number": sum(m["val/operation_number"] * m["val/n_episode"] for m in metrics_list) / n_episode,
                "val/n_episode": n_episode,
            })
            metrics["val/shortage_percentage"] = metrics["val/container_shortage"] / metrics["val/order_requirements"]

        return metrics
